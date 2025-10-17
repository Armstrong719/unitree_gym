
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import time
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1ULCRobot(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.n_proprio = cfg.env.n_proprio
        self.history_len = cfg.env.history_len

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs , device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.debug_viz = False

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_vec = torch.zeros_like(self.obs_buf[0, :self.cfg.env.n_proprio])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:27] = 0. # commands
        noise_vec[27:27+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[27+self.num_actions:27+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[27+2*self.num_actions:27+3*self.num_actions] = 0. # previous actions
        noise_vec[27+3*self.num_actions:27+3*self.num_actions+2] = 0. # contact
        
        return noise_vec

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # self.actions[:, 12:] = 0.
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        alpha = torch.concat([self.alpha_1, self.alpha_2, self.alpha_3])
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, alpha
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.target_pos += self.commands[:, :2]*self.dt
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # compute current com 
        for i in range(self.num_dofs+1):
            self.mid_rigid_body_states_local[:, i, :] = self.rigid_body_states_view[:, i, :3] - self.root_states[:, 0:3] + quat_rotate(self.rigid_body_states_view[:, i, 3:7], self.com_local_pos[:, i, :])
            self.rigid_body_states_local[:, i, :] = quat_apply(quat_conjugate(self.root_states[:, 3:7]), self.mid_rigid_body_states_local[:, i, :]) # rigid body pos in base frame
        self.body_com = (self.rigid_body_states_local[:, :, :3] * self.masses.unsqueeze(2)).sum(dim=1)/self.masses.sum(dim=1).unsqueeze(1)
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        distance_error = torch.norm(self.target_pos - self.base_pos[:, :2], dim=1)
        distance_buf = distance_error >= 0.5 # reset if the base pos is 0.5m far from target
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= distance_buf

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.rigid_body_states_local = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.mid_rigid_body_states_local = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        if self.cfg.env.history_len > 0:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)
            self.privileged_obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, device=self.device, dtype=torch.float)

        # arms random variable
        self.interp_mode = torch.rand(self.num_envs, dtype=torch.float32, device=self.device)
        self.random_motion_period = torch.ones((self.num_envs,1), dtype=torch.float, device=self.device, requires_grad=False)*0.5
        self.last_target_arm_dof_pos = torch.zeros((self.num_envs, 14), dtype=torch.float, device=self.device, requires_grad=False)
        self.target_arm_dof_pos = torch.zeros((self.num_envs, 14), dtype=torch.float, device=self.device, requires_grad=False)
        self.mid_point = torch.zeros((self.num_envs, 14), dtype=torch.float, device=self.device, requires_grad=False)
        self.amplitude = torch.zeros((self.num_envs, 14), dtype=torch.float, device=self.device, requires_grad=False)
        self.current_action = torch.zeros((self.num_envs, 14), dtype=torch.float, device=self.device, requires_grad=False)
        self.current_time = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device, requires_grad=False)
        self.start_time = torch.zeros((self.num_envs, 1), dtype=torch.float, device=self.device, requires_grad=False)

        # commands variable
        self.motion_flag = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.avgrew_velocity = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.avgrew_height = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.avgrew_hip = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.avgrew_upper = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.avgrew_torso = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.alpha_1 = torch.ones(1, device=self.device, dtype=torch.float32)*0.05
        self.alpha_2 = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.alpha_3 = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.target_pos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)



      

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self._init_foot()

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        self.masses = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.float, device=self.device)
        self.com_local_pos = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.body_com = torch.zeros(self.num_envs)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        torso_names = [s for s in body_names if self.cfg.asset.torso_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.masses[i] = torch.tensor([prop.mass for prop in body_props], device=self.device)
            self.com_local_pos[i] = torch.tensor([[prop.com.x, prop.com.y, prop.com.z] for prop in body_props], device=self.device)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.torso_indices = torch.zeros(len(torso_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(torso_names)):
            self.torso_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], torso_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()
        if self.common_step_counter % 150 == 0: 
            self.update_adaptive_curriculum()

        period = 1.
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = torch.where(self.motion_flag == 0, 0, self.phase)
        self.phase_right = torch.where(self.motion_flag == 0, 0, (self.phase + offset) % 1)
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        self.contact_sequence = 1*(self.leg_phase < 0.55)
        
        return super()._post_physics_step_callback()
    
    def update_adaptive_curriculum(self):
        # update period 3s
        
        C_vel = self.avgrew_velocity >= 0.8
        C_hgt = self.avgrew_height >= 0.85
        C_hip = self.avgrew_hip <= 0.2
        C_upper = self.avgrew_upper >= 0.7
        C_torso = self.avgrew_torso >= 0.8
        C_complete = self.alpha_3 >= 0.98

        Cu_1 = C_vel
        Cu_2 = C_hgt & C_vel 
        Cu_3 = C_upper & C_torso & Cu_2 & C_complete

        # print("vel:", self.avgrew_velocity)
        # print("hgt:", self.avgrew_height)
        # print("alpha2:", self.alpha_2)
        # print("--"*10)


        # if Cu_1 and (self.alpha_1 < 0.98):
        #     self.alpha_1 = torch.clip(self.alpha_1 + 0.05, max=0.98)
        if Cu_2 and (self.alpha_2 < 0.98):
            self.alpha_2 = torch.clip(self.alpha_2 + 0.05, max=0.98)
        if Cu_3 and (self.alpha_3 < 0.98):
            self.alpha_3 = torch.clip(self.alpha_3 + 0.05, max=0.98)


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed

        Structure:
            cmd = [cmd_loco, cmd_torso, cmd_arms]
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["height"][0] + (1-self.alpha_2)*(self.command_ranges["height"][1] - self.command_ranges["height"][0]), \
                                                     self.command_ranges["height"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["torso_yaw"][0] + (1-self.alpha_3)*(self.command_ranges["torso_yaw"][1] - self.command_ranges["torso_yaw"][0]), \
                                                     self.command_ranges["torso_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 5] = torch_rand_float(self.command_ranges["torso_roll"][0] + (1-self.alpha_3)*(self.command_ranges["torso_roll"][1] - self.command_ranges["torso_roll"][0]), \
                                                     self.command_ranges["torso_roll"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 6] = torch_rand_float(self.command_ranges["torso_pitch"][0] + (1-self.alpha_3)*(self.command_ranges["torso_pitch"][1] - self.command_ranges["torso_pitch"][0]), \
                                                     self.command_ranges["torso_pitch"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 7:] = self.random_arm_dof_pos()[env_ids, :]*self.alpha_3

        # set some reset env to stand
        rand_idx = (torch.rand(len(env_ids), device=self.device) < self.cfg.commands.stand_ratio) # 50% reset env stand
        zero_idx = env_ids[rand_idx] # resampled envs of stand
        non_zero_idx = env_ids[~rand_idx] # resampled envs of walk
        self.commands[zero_idx, :3] = 0.      # set all standing vel to be 0.

        self.motion_flag[zero_idx] = 0 # 0 for stand, 1 for walk
        self.motion_flag[non_zero_idx] = 1





    
    def random_arm_dof_pos(self):
        def biased_distribution_sampling(lower, upper, mean, std, size, type, device):
            if type == 'Normal':
                batch = torch.normal(mean=mean, std=std, size=(size, ), device=device)
                clipped_batch = torch.clip(batch, min=lower, max=upper)
                return clipped_batch

            elif type == 'Uniform':
                batch = lower + (upper-lower)*torch.rand(size, device=device)
                return batch


        # get the resample idx, resample period, target dof pos, last_target_dof_pos, motion mode
        self.current_time = torch.round(self.episode_length_buf * self.dt, decimals=2).unsqueeze(1)
        resample_idx = ((self.current_time-self.start_time) >= self.random_motion_period).nonzero(as_tuple=False).flatten()
        # breakpoint()
        self.start_time[resample_idx] = self.current_time[resample_idx]
        # breakpoint()
        self.interp_mode[resample_idx] = torch.rand(len(resample_idx), device=self.device)
        sin_wave_idx = (self.interp_mode > 0.5).nonzero(as_tuple=False).flatten()
        ramp_idx = (self.interp_mode <= 0.5).nonzero(as_tuple=False).flatten()
        self.random_motion_period[resample_idx] = torch_rand_float(lower=1., upper=3., shape=(len(resample_idx), 1), device=self.device)
        # self.current_time_period[resample_idx] = torch.round((self.episode_length_buf[resample_idx] * self.dt) % self.random_motion_period[resample_idx], decimals=2)
        self.last_target_arm_dof_pos[resample_idx] = self.current_action[resample_idx]
        dist_type = 'Uniform'
        arm_pos_curriculum = self.episode_length_buf.unsqueeze(1)/4000

        self.target_arm_dof_pos[resample_idx, 0] = biased_distribution_sampling(self.dof_pos_limits[15, 0], self.dof_pos_limits[15, 1], 0., 1.5, len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 1] = biased_distribution_sampling(self.dof_pos_limits[16, 0], self.dof_pos_limits[16, 1], -2., 1.5, len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 2] = biased_distribution_sampling(self.dof_pos_limits[17, 0], self.dof_pos_limits[17, 1], 1., 1.5, len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 3] = biased_distribution_sampling(self.dof_pos_limits[18, 0], self.dof_pos_limits[18, 1], 0., 1., len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 4] = biased_distribution_sampling(self.dof_pos_limits[19, 0], self.dof_pos_limits[19, 1], 0., 0.8, len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 5] = biased_distribution_sampling(self.dof_pos_limits[20, 0], self.dof_pos_limits[20, 1], 0., 1., len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 6] = biased_distribution_sampling(self.dof_pos_limits[21, 0], self.dof_pos_limits[21, 1], 0., 1.0, len(resample_idx), dist_type, device=self.device)
        
        self.target_arm_dof_pos[resample_idx, 7] = biased_distribution_sampling(self.dof_pos_limits[22, 0], self.dof_pos_limits[22, 1], -2., 1.5, len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 8] = biased_distribution_sampling(self.dof_pos_limits[23, 0], self.dof_pos_limits[23, 1], -1., 1.5, len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 9] = biased_distribution_sampling(self.dof_pos_limits[24, 0], self.dof_pos_limits[24, 1], 0., 1., len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 10] = biased_distribution_sampling(self.dof_pos_limits[25, 0], self.dof_pos_limits[25, 1], 0., 0.8, len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 11] = biased_distribution_sampling(self.dof_pos_limits[26, 0], self.dof_pos_limits[26, 1], 0., 1., len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 12] = biased_distribution_sampling(self.dof_pos_limits[27, 0], self.dof_pos_limits[27, 1], 0., 1.0, len(resample_idx), dist_type, device=self.device)
        self.target_arm_dof_pos[resample_idx, 13] = biased_distribution_sampling(self.dof_pos_limits[28, 0], self.dof_pos_limits[28, 1], 0., 1.0, len(resample_idx), dist_type, device=self.device)
        
        # self.target_arm_dof_pos[resample_idx, :] *= arm_pos_curriculum[resample_idx, :]
        cur_time_expand = self.current_time.expand(self.num_envs, 14)
        rand_period_expand = self.random_motion_period.expand(self.num_envs, 14)
        periodic_time = cur_time_expand - self.start_time
        # interpolate between last target pos and current target pos with two modes, sin and ramp
        pos_idx = self.target_arm_dof_pos >= self.last_target_arm_dof_pos
        #ramp function, after 0.5s, stay static at target pos
        self.current_action[ramp_idx] = self.last_target_arm_dof_pos[ramp_idx] + \
                                            (self.target_arm_dof_pos[ramp_idx] - self.last_target_arm_dof_pos[ramp_idx])*torch.clip(periodic_time[ramp_idx], max=1.)/1.

        self.mid_point[sin_wave_idx] = 0.5*(self.target_arm_dof_pos[sin_wave_idx] + self.last_target_arm_dof_pos[sin_wave_idx])
        self.amplitude[sin_wave_idx] = 0.5*(self.target_arm_dof_pos[sin_wave_idx] - self.last_target_arm_dof_pos[sin_wave_idx])

        self.current_action[sin_wave_idx] = self.mid_point[sin_wave_idx]-self.amplitude[sin_wave_idx]*torch.cos(2*torch.pi*periodic_time[sin_wave_idx]/(2*rand_period_expand[sin_wave_idx]))

        return self.current_action

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.current_action[env_ids, :] = 0.
        self.target_arm_dof_pos[env_ids, :] = 0.
        self.last_target_arm_dof_pos[env_ids, :] =0.
        self.random_motion_period[env_ids, :]=torch.ones((len(env_ids),1), dtype=torch.float, device=self.device, requires_grad=False)*1.
        self.current_time[env_ids] = 0.
        self.start_time[env_ids] = 0.
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        self.target_pos[env_ids, :] = self.root_states[env_ids, :2] # global frame
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.contact_sequence
                                    ),dim=-1)
        # print(self.contact_sequence)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        if self.cfg.env.history_len > 0:
            self.obs_buf = torch.cat([self.obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([self.obs_buf[:, :self.cfg.env.n_proprio]] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.obs_history_buf[:, 1:],
                    self.obs_buf[:, :self.cfg.env.n_proprio].unsqueeze(1)
                ], dim=1))
            

        self.privileged_obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.contact_sequence
                                    ),dim=-1)
        if self.cfg.env.history_len > 0:
            self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, self.privileged_obs_history_buf.view(self.num_envs, -1)], dim=-1)
            self.privileged_obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([self.privileged_obs_buf[:, :self.cfg.env.n_proprio]] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.privileged_obs_history_buf[:, 1:],
                    self.privileged_obs_buf[:, :self.cfg.env.n_proprio].unsqueeze(1)
                ], dim=1))
        # add perceptive inputs if not blind
        # add noise if needed

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        self.gym.clear_lines(self.viewer)
        sphere_geom_current_pos = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0.7, 0.5, 0))
        sphere_geom_target_pos = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0., 0.5, 0.7))
        for i in range(self.num_envs):
            # visualize robot height and target height


            current_pos = (self.root_states[i, :3]).cpu().numpy()
            target_pos = self.target_pos[i].cpu().numpy()
            
            # com_world = self.body_com[i, :3].cpu().numpy() + base_pos
            # cop_world = self.body_cop[i, :3].cpu().numpy() + base_pos
            cur_x = current_pos[0]
            cur_y = current_pos[1] 
            cur_z = 0.01
            tar_x = target_pos[0]
            tar_y = target_pos[1]
            tar_z = 0.01

            # print('visual dist',np.sqrt((com_x - lft_x)**2 + (com_y - lft_y)**2))
            # print('visual com', com_world)
            # print('visual feet',feet_world )
            current_sphere_pose = gymapi.Transform(gymapi.Vec3(cur_x, cur_y, cur_z), r=None)
            target_sphere_pose = gymapi.Transform(gymapi.Vec3(tar_x, tar_y, tar_z), r=None)
            gymutil.draw_lines(sphere_geom_current_pos, self.gym, self.viewer, self.envs[i], current_sphere_pose)
            gymutil.draw_lines(sphere_geom_target_pos, self.gym, self.viewer, self.envs[i], target_sphere_pose)



    def _reward_termination(self):
        # Terminal reward / penalty
        return -200*self.reset_buf * ~self.time_out_buf
        
    # def _reward_contact(self):
    #     res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    #     for i in range(self.feet_num):
    #         is_stance = self.leg_phase[:, i] < 0.55
    #         contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
    #         res += ~(contact ^ is_stance)
    #     # print("target contact", is_stance)
    #     # print("real contact", contact)
    #     # print("--"*10)
    #     return res
    
    def _reward_contact_tracking(self):

        real_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        hard_sequence = torch.where(self.contact_sequence > 0, torch.ones_like(self.contact_sequence), torch.zeros_like(self.contact_sequence))

        correct_contact = (real_contact == hard_sequence)
        raw_rew = 0.5*torch.sum(correct_contact * 1, dim = 1)
        error = 1 - raw_rew
        # print('seq', hard_sequence)
        # print('real', real_contact)
        # print('correct', correct_contact)
        # print('rew', torch.exp(-error*2))
        return torch.exp(-error*2)
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_action_diff(self):
        return torch.exp(-0.05 * (torch.norm(self.last_actions[:, :12] - self.actions[:, :12], dim=-1)))
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        self.avgrew_velocity = torch.mean(torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma))
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_height(self):
        height_error = torch.square(self.commands[:, 3] - self.root_states[:, 2])
        self.avgrew_height = torch.mean(torch.exp(-height_error/self.cfg.rewards.tracking_sigma))
        return torch.exp(-height_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_upperbody_pos(self):
        upperbody_pos_error = torch.sum(torch.square(self.commands[:, 7:] - self.dof_pos[:, 15:]), dim=1)
        self.avgrew_upper = torch.mean(torch.exp(-upperbody_pos_error/0.35**2))
        return torch.exp(-upperbody_pos_error/0.35**2)
    
    def _reward_tracking_torso_yaw(self):
        local_torso_orient = quat_mul(quat_conjugate(self.root_states[:, 3:7]), self.rigid_body_states_view[:, self.torso_indices, 3:7].squeeze(1))
        _, _, raw_yaw = get_euler_xyz(local_torso_orient) # yaw respect to pelvis
        raw_roll, raw_pitch, _ = get_euler_xyz(self.rigid_body_states_view[:, self.torso_indices, 3:7].squeeze(1)) # roll pitch respect to global(gravity)
        base_roll = torch.where(torch.logical_and(raw_roll<=torch.pi, raw_roll>=-torch.pi), raw_roll,
                    torch.where(raw_roll > torch.pi, raw_roll - 2*torch.pi,
                    torch.where(raw_roll < -torch.pi, raw_roll + 2*torch.pi, raw_roll)))
        
        base_pitch = torch.where(torch.logical_and(raw_pitch<=torch.pi, raw_pitch>=-torch.pi), raw_pitch,
                    torch.where(raw_pitch > torch.pi, raw_pitch - 2*torch.pi,
                    torch.where(raw_pitch < -torch.pi, raw_pitch + 2*torch.pi, raw_pitch)))
        
        base_yaw = torch.where(torch.logical_and(raw_yaw<=torch.pi, raw_yaw>=-torch.pi), raw_yaw,
                    torch.where(raw_yaw > torch.pi, raw_yaw - 2*torch.pi,
                    torch.where(raw_yaw < -torch.pi, raw_yaw + 2*torch.pi, raw_yaw)))
        
        torso_roll_error = torch.square(self.commands[:, 5] - base_roll)
        torso_pitch_error = torch.square(self.commands[:, 6] - base_pitch)
        torso_yaw_error = torch.square(self.commands[:, 4] - base_yaw)
        # print("roll", base_roll)
        # print("pitch", base_pitch)
        # print("yaw", base_yaw)
        # print("-"*20)
        self.avgrew_torso = torch.mean(0.25*torch.exp(-torso_yaw_error/0.2**2) + 0.25*torch.exp(-torso_roll_error/0.2**2) + \
                                       0.5*torch.exp(-torso_pitch_error/0.2**2))
        # print("base yaw", base_yaw)
        # print("torso yaw error", torso_yaw_error)
        # print("yaw command", self.commands[:, 4])
        return torch.exp(-torso_yaw_error/0.2**2)

    def _reward_tracking_torso_roll(self):
        raw_roll, _, _ = get_euler_xyz(self.rigid_body_states_view[:, self.torso_indices, 3:7].squeeze(1))
        base_roll = torch.where(torch.logical_and(raw_roll<=torch.pi, raw_roll>=-torch.pi), raw_roll,
                    torch.where(raw_roll > torch.pi, raw_roll - 2*torch.pi,
                    torch.where(raw_roll < -torch.pi, raw_roll + 2*torch.pi, raw_roll)))
        
        torso_roll_error = torch.square(self.commands[:, 5] - base_roll)
        return torch.exp(-torso_roll_error/0.2**2)

    def _reward_tracking_torso_pitch(self):
        _, raw_pitch, _ = get_euler_xyz(self.rigid_body_states_view[:, self.torso_indices, 3:7].squeeze(1))
        base_pitch = torch.where(torch.logical_and(raw_pitch<=torch.pi, raw_pitch>=-torch.pi), raw_pitch,
                    torch.where(raw_pitch > torch.pi, raw_pitch - 2*torch.pi,
                    torch.where(raw_pitch < -torch.pi, raw_pitch + 2*torch.pi, raw_pitch)))
        
        torso_pitch_error = torch.square(self.commands[:, 6] - base_pitch)
        return torch.exp(-torso_pitch_error/0.2**2)

    def _reward_tracking_cog(self):
        base_pos = (self.root_states[:, :3])
        com_world = quat_apply(self.root_states[:, 3:7], self.body_com[:, :3]) + base_pos
        feet_world = quat_apply(self.root_states[:, 3:7], torch.mean(self.rigid_body_states_local[:, self.feet_indices, :3], dim=1)) + base_pos
        dist_com = torch.sum(torch.square(feet_world[:, :2] - com_world[:, :2]), dim=1)
        return torch.exp(-dist_com/0.2**2)

    def _reward_z_lin_vel(self):
        return -1*torch.square(self.base_lin_vel[:, 2])

    def _reward_energy(self):
        energy_consum = torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)
        return -0.001*energy_consum
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_torso_dof_vel(self):
        # Penalize dof velocities
        # print(torch.sum(torch.square(self.dof_vel[:, 12:15]), dim=1))
        return torch.sum(torch.square(self.dof_vel[:, 12:15]), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        # print("waist action diff", torch.sum(torch.square(self.last_actions[:, 12:15] - self.actions[:, 12:15]), dim=1))
        # print("waist dof pos", self.dof_pos[:, 12:15])
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_base_orientation(self):
        return -5*torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) # ?????

    def _reward_joint_pos_limit(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return -2.*torch.sum(out_of_limits, dim=1)

    def _reward_joint_effort_limit(self):
        # penalize torques too close to the limit
        return -2*torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)


    def _reward_joint_deviation(self):
        pitch_idx = [0, 3, 4, 6, 9, 10]
        hip_yaw_ankle_idx = [2, 5, 8, 11]
        hip_roll_idx = [1, 7]
        dev1 = torch.sum(torch.abs(self.dof_pos[:, hip_yaw_ankle_idx] - self.default_dof_pos[0, hip_yaw_ankle_idx]), dim=1)
        dev2 = torch.sum(torch.abs(self.dof_pos[:, hip_roll_idx] - self.default_dof_pos[0, hip_roll_idx]), dim=1)
        pitch_dev = torch.sum(torch.abs(self.dof_pos[:, pitch_idx] - self.default_dof_pos[0, pitch_idx]), dim=1)

        self.avgrew_hip = torch.mean(pitch_dev)
        return -0.15*dev1 -0.3*dev2 -0.15*pitch_dev

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime = torch.sum(torch.clamp(self.feet_air_time, max=0.4), dim=1)
        rew_airTime *= self.motion_flag == 1 #no reward for stand
        self.feet_air_time *= ~contact_filt
        return 0.3*rew_airTime
    
    def _reward_feet_slide(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        foot_speed_norm = torch.norm(self.rigid_body_states_view[:, self.feet_indices, 7:9], dim=-1) # feet xy vel
        rew = foot_speed_norm * contact
        # print(rew)
        return -0.25*torch.sum(rew, dim=1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return -3e-3*torch.sum((self.contact_forces[:, self.feet_indices, 2] -  self.cfg.rewards.max_contact_force).clip(min=0.).clip(max=400), dim=1)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return -2.0*(torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1))

    def _reward_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1. * contacts, dim=1) >= 1
        fly = torch.sum(1. * contacts, dim=1) < 1
        # print(fly)
        return -1. * fly

    def _reward_undesired_contact(self):

        rew = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1., dim=1)
        return -1*rew

    def _reward_ankle_orientation(self):
        left_ankle_projected_gravity = quat_rotate_inverse(self.feet_state[:, 0, 3:7], self.gravity_vec)
        right_ankle_projected_gravity = quat_rotate_inverse(self.feet_state[:, 0, 3:7], self.gravity_vec)
        rew = torch.sum(torch.square(left_ankle_projected_gravity[:, :2]), dim=-1) + torch.sum(torch.square(right_ankle_projected_gravity[:, :2]), dim=-1)
        return -0.5*rew
    
    


    
    
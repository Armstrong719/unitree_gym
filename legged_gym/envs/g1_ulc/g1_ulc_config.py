from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1ULCRoughCfg( LeggedRobotCfg ):
    class env:
        num_envs = 4096
        num_observations = 116 + 116*6
        n_proprio = 116
        history_len = 6
        num_privileged_obs = 116 + 116*6 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 29
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        test = False

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       

           'waist_yaw_joint' : 0.,
           'waist_roll_joint' : 0.,
           'waist_pitch_joint' : 0.,

           'left_shoulder_pitch_joint' : 0.,
           'left_shoulder_roll_joint' : 0.,
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint' : 0.,
           'left_wrist_roll_joint' : 0.,
           'left_wrist_pitch_joint' : 0.,
           'left_wrist_yaw_joint' : 0.,

           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
           'right_wrist_roll_joint' : 0.,
           'right_wrist_pitch_joint' : 0.,
           'right_wrist_yaw_joint' : 0.,

        }
    


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 21 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, height, torso yaw, torso roll, torso pitch, arms pos(14)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        stand_ratio = 1
        class ranges:
            lin_vel_x = [-0.45, 0.55] # min max [m/s]
            lin_vel_y = [-0.45, 0.45]   # min max [m/s]
            ang_vel_yaw = [-1.2, 1.2]    # min max [rad/s]
            height = [0.3, 0.75]      
            # torso_yaw = [-2.62, 2.62]
            # torso_roll = [-0.52, 0.52]
            # torso_pitch = [-0.52, 1.57]
            torso_yaw = [-0., 0.]
            torso_roll = [-0.0, 0.0]
            torso_pitch = [-0.0, 0.]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 200,
                     'ankle_pitch': 20,
                     'ankle_roll': 20,

                     'waist_yaw':400,
                     'waist_roll':400,
                     'waist_pitch':400,

                     'shoulder_pitch': 90,
                     'shoulder_roll': 60,
                     'shoulder_yaw': 20,
                     'elbow': 60,
                     'wrist_roll': 20,
                     'wrist_pitch': 20,
                     'wrist_yaw': 20,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2.5,
                     'hip_roll': 2.5,
                     'hip_pitch': 2.5,
                     'knee': 5,
                     'ankle_pitch': 0.2,
                     'ankle_roll': 0.1,

                     'waist_yaw':5,
                     'waist_roll':5,
                     'waist_pitch':5,

                     'shoulder_pitch': 2,
                     'shoulder_roll': 1,
                     'shoulder_yaw': 0.4,
                     'elbow': 1,
                     'wrist_roll': 0.4,
                     'wrist_pitch': 0.4,
                     'wrist_yaw': 0.4,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # stiffness = {'hip_yaw': 100,
        #              'hip_roll': 100,
        #              'hip_pitch': 100,
        #              'knee': 150,
        #              'ankle': 40,

        #              'waist_yaw':10,
        #              'waist_roll':10,
        #              'waist_pitch':10,

        #              'shoulder_pitch': 10,
        #              'shoulder_roll': 10,
        #              'shoulder_yaw': 1,
        #              'elbow': 10,
        #              'wrist_roll': 10,
        #              'wrist_pitch': 10,
        #              'wrist_yaw': 10,
        #              }  # [N*m/rad]
        # damping = {  'hip_yaw': 2,
        #              'hip_roll': 2,
        #              'hip_pitch': 2,
        #              'knee': 4,
        #              'ankle': 2,

        #              'waist_yaw':1,
        #              'waist_roll':1,
        #              'waist_pitch':1,

        #              'shoulder_pitch': 1,
        #              'shoulder_roll': 1,
        #              'shoulder_yaw': 1,
        #              'elbow': 1,
        #              'wrist_roll': 1,
        #              'wrist_pitch': 1,
        #              'wrist_yaw': 1,
        #              }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_rev_1_0.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        torso_name = "torso"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis", "hip"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.999
        max_contact_force = 500. # forces above this value are penalized
        
        class scales( LeggedRobotCfg.rewards.scales ):
            # tracking_lin_vel = 2.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # orientation = -1.0
            # # base_height = -10.0
            # dof_acc = -2.5e-7
            dof_vel = -1e-4
            torso_dof_vel = -1e-2
            # feet_air_time = 0.0
            # collision = 0.0
            # action_rate = -0.01
            # dof_pos_limits = -5.0
            alive = 1.
            # hip_pos = -1.0
            # contact_no_vel = -0.2
            # feet_swing_height = -20.0
            contact = 1.
            action_diff = 2.

            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.25
            tracking_height = 1.0
            tracking_upperbody_pos = 2.0
            tracking_torso_yaw = 1
            tracking_torso_roll = 1
            tracking_torso_pitch = 1
            tracking_cog = 1

            # termination = 1.0
            z_lin_vel = 1.0
            energy = 1.0
            dof_acc = 1.0
            action_rate = 1.0
            base_orientation = 1.0
            joint_pos_limit = 1.0
            joint_effort_limit = 1.0
            joint_deviation = 1.0
            feet_air_time = 1.0
            feet_slide = 1.0
            feet_contact_forces = 1.0
            stumble = 1.0
            fly = 1.0
            undesired_contact = 1.0
            ankle_orientation = 1.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            # lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            # height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class G1ULCRoughCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [1024, 512, 512, 256]
        critic_hidden_dims = [1024, 512, 512, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 64
        # rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.006
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic"
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 10000

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'g1_ulc'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

  

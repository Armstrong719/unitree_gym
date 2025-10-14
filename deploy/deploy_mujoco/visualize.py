import mujoco
import mujoco.viewer
import numpy as np
import keyboard 
import time


def analyze_joints(model_path):
    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # print("=" * 60)
    # print("JOINT ANALYSIS REPORT")
    # print("=" * 60)
    
    # # 1. 基本信息
    # print(f"\n模型基本信息:")
    # print(f"关节数量 (nq): {model.nq}")
    # print(f"执行器数量 (nu): {model.nu}")
    # print(f"物体数量 (nbody): {model.nbody}")
    
    # # 2. 关节详细信息
    # print(f"\n关节详细信息:")
    # for i in range(model.njnt):
    #     jnt_type = model.jnt_type[i]
    #     jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    #     jnt_qposadr = model.jnt_qposadr[i]
    #     jnt_dofadr = model.jnt_dofadr[i]
        
    #     type_str = {
    #         0: "自由关节",
    #         1: "球关节", 
    #         2: "滑动关节",
    #         3: "铰链关节"
    #     }.get(jnt_type, "未知")
        
    #     print(f"关节 {i}: '{jnt_name}'")
    #     print(f"  类型: {type_str} ({jnt_type})")
    #     print(f"  qpos地址: {jnt_qposadr}")
    #     print(f"  dof地址: {jnt_dofadr}")
        
    #     # 关节范围
    #     if model.jnt_limited[i]:
    #         range_min = model.jnt_range[i][0]
    #         range_max = model.jnt_range[i][1]
    #         print(f"  范围: [{range_min:.3f}, {range_max:.3f}]")
    #     else:
    #         print(f"  范围: 无限制")
            
    #     # 初始位置
    #     if jnt_qposadr >= 0:
    #         qpos_val = data.qpos[jnt_qposadr] if jnt_qposadr < model.nq else 0
    #         print(f"  初始位置: {qpos_val:.3f}")
    
    # # 3. 执行器信息
    # print(f"\n执行器信息:")
    # for i in range(model.nu):
    #     act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    #     act_joint = model.actuator_trnid[i][0]
    #     joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, act_joint)
        
    #     print(f"执行器 {i}: '{act_name}' -> 关节 '{joint_name}'")
        
    #     # 执行器控制范围
    #     if model.actuator_ctrllimited[i]:
    #         ctrl_range_min = model.actuator_ctrlrange[i][0]
    #         ctrl_range_max = model.actuator_ctrlrange[i][1]
    #         print(f"  控制范围: [{ctrl_range_min:.3f}, {ctrl_range_max:.3f}]")
    
    # # 4. 身体(连杆)信息
    # print(f"\n身体(连杆)信息:")
    # for i in range(1, model.nbody):  # 从1开始跳过世界体
    #     body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    #     body_pos = model.body_pos[i]
    #     body_quat = model.body_quat[i]
        
    #     print(f"身体 {i}: '{body_name}'")
    #     print(f"  位置: [{body_pos[0]:.3f}, {body_pos[1]:.3f}, {body_pos[2]:.3f}]")
    
    return model, data

def visualize_with_controls(model_path):
    """可视化并允许控制每个关节"""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # print(f"\n控制说明:")
    # print("- 按 ']' 键切换到下一个关节")
    # print("- 按 '[' 键切换到上一个关节") 
    # print("- 按 '↑' 键增加关节位置")
    # print("- 按 '↓' 键减少关节位置")
    # print("- 当前控制的关节会显示在窗口标题中")
    
    current_joint = 0
    pressed_flag = False
    
    def key_callback(key):
        return
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if keyboard.is_pressed('right') and not pressed_flag:
                current_joint = np.clip((current_joint + 1), 0, model.njnt)
                pressed_flag = True
            elif keyboard.is_pressed('left') and not pressed_flag:
                current_joint = np.clip((current_joint - 1), 0, model.njnt)
                pressed_flag = True
            elif not keyboard.is_pressed('right') and not keyboard.is_pressed('left'):
                pressed_flag = False




            data.qpos[:current_joint] = 0
            data.qpos[current_joint+1:] = 0
            data.qvel[:] = 0
            # 获取按键状态
            
            if keyboard.is_pressed('up'):  # 上箭头
                data.qpos[current_joint] += 0.1
            elif keyboard.is_pressed('down'):  # 下箭头
                data.qpos[current_joint] -= 0.1
            data.qpos[current_joint] = np.clip(data.qpos[current_joint], model.jnt_range[current_joint][0], model.jnt_range[current_joint][1])

            print('current joint', current_joint)
            print('current_pos', data.qpos[current_joint])
            print('joint_limits', model.jnt_range[current_joint])
            print('-'*20)

            time.sleep(0.01)

            
            # 显示当前控制的关节
            # joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, current_joint)
            # viewer.user_scn.ntitle = f"当前控制: {joint_name} (关节 {current_joint})"
            
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    model_path = "resources/robots/g1_description/g1_29dof_rev_1_0.urdf"
    
    # 分析关节信息
    model, data = analyze_joints(model_path)
    
    # 可视化并交互控制
    visualize_with_controls(model_path)
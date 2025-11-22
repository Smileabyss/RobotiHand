
def pd_control(target_q, q, kp, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp  - dq * kd

def run(env):
    # 1. 加载模型并移至设备（与环境保持一致）
    policy = torch.jit.load("/home/cyrus/ManipTrans/Proprioception_envs/model/policy_29dof.pt").to(env.cfg.DEVICE)
    policy.eval()  # 推理模式，禁用随机化
    

    
    # 3. 控制频率参数（与配置文件一致：control_decimation=7）
    control_decimation = 7  # 每7个仿真步执行一次控制更新
    sim_counter = 0  # 仿真步计数器
    
    # 4. 初始化参数
    cmd = [0.0, 0.0, 0.0]  # 初始指令（静止）
    batch_size = env.cfg.envs_num  # 环境数量（如4）
    device = env.cfg.DEVICE
    
    # 初始化动作（模型输出的原始动作，按模型关节顺序）
    init_action = torch.zeros((batch_size, env.dof_count), dtype=torch.float32, device=device)
    old_action = init_action  # 上一时刻动作（模型顺序，用于观测）
    
    # PD参数（批量适配，与关节数匹配）
    kp = torch.tensor(env.cfg.kps, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
    kd = torch.tensor(env.cfg.kds, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    # 关节映射表（模型顺序→电机关节顺序）
    joint2motor_idx = [0, 6, 12, 
                       1, 7, 13, 
                       2, 8, 14, 
                       3, 9, 15, 22,
                       4, 10, 16, 23, 
                       5, 11, 17, 24, 
                       18, 25, 
                       19, 26, 
                       20, 27, 
                       21, 28]
    
    # 5. 重置环境并获取初始观测
    obs = env.reset_env()
    
    # 初始化力矩（首次控制前用零力矩）
    torques = torch.zeros((batch_size, env.dof_count), dtype=torch.float32, device=device)
    
    while True:
        # 6. 控制降采样：每7个仿真步执行一次模型推理和动作更新
        if sim_counter % control_decimation == 0:
            # 6.1 构建模型输入（按模型要求的关节顺序和缩放）
            states = computer_observation(obs, cmd, old_action, env.cfg.default_angles)
            
            # 6.2 模型推理（启用推理模式，提升性能和稳定性）
            action = policy(states.clip(-100, 100)).clip(-10, 10)  # 模型输出（模型关节顺序）
            
            # 6.3 动作映射：模型输出→目标关节角度（带限位保护）
            default_angle_tensor = torch.tensor(
                env.cfg.default_angles, dtype=torch.float32, device=device
            ).unsqueeze(0).repeat(batch_size, 1)
            dof_action = action * 0.25 + default_angle_tensor  # 目标角度（模型顺序）
            
                        
            # 6.4 动作重排：模型顺序→电机关节顺序（仅下发时重排）
            action_reorder = dof_action.clone()
            kps_re = kp.clone()
            kds_re = kd.clone()
            for i in range(len(joint2motor_idx)):
                motor_idx = joint2motor_idx[i]
                action_reorder[:, motor_idx] = dof_action[:, i]  # 映射到电机实际顺序
                kps_re[:,motor_idx] = kp[:,i]
                kds_re[:,motor_idx] = kd[:,i]
            
            # 6.5 获取当前关节状态（位置q、速度dq，电机顺序）
            q, dq, _, _, _ = env._read_state()  # q和dq需与action_reorder顺序一致

            angle_error = dof_action - q  # 模型顺序的误差
            print(f"关节角度误差范围：[{angle_error.min():.2f}, {angle_error.max():.2f}]")
            
            # 6.6 PD计算力矩 + 力矩限制（防止过大力矩导致抖动）
            torques = pd_control(dof_action, q, kps_re, dq, kds_re)
 
            # x = torques[0,0].clone()
            # torques[0,0]=torques[0,2]
            # torques[0,2] =  x

            # x = torques[0,6].clone()
            # torques[0,6]=torques[0,8]
            # torques[0,8] =  x

            # 6.7 更新上一时刻动作（模型顺序，用于下一帧观测）
            old_action = action  # 注意：用模型顺序的动作，而非重排后的
            
            # 6.8 调试：每10个控制帧打印一次信息（确认频率是否正确）
            if sim_counter % (control_decimation * 10) == 0:
                print(f"控制帧：{sim_counter//control_decimation}，力矩范围：[{torques.min():.2f}, {torques.max():.2f}]")
        
        # 7. 执行仿真步（无论是否为控制帧，均推进仿真）
        # 非控制帧时，复用上次计算的torques，避免动作频繁变化
        observation, done = env.step(torques)
        
        # 8. 仅在控制帧更新观测（非控制帧观测不变，模型输入稳定）
        if sim_counter % control_decimation == 0:
            obs = observation  # 更新观测为最新状态
        
        # 9. 递增仿真计数器
        sim_counter += 1

# 修正1：函数定义删除多余的env参数，参数顺序与调用一致
def computer_observation(obs, cmd, old_action, default_angle):
    envs_num = obs.shape[0]  
    device = obs.device  # 从obs获取设备，确保一致性（正确）
    
    # 1. 观测拆分（按拼接顺序反向拆分，维度计算正确）
    base_ang_vel = obs[:, 0:3]  # 3维（正确）
    base_gravity_orientation = obs[:, 3:6]  # 3维（正确）
    dof_pos = obs[:, 6:35]  # 29维（6~35：35-6=29，正确）
    dof_vel = obs[:, 35:64]  # 29维（35~64：64-35=29，正确）
    
    # 2. dof_pos处理（减去默认角度，逻辑正确）
    default_angles = torch.tensor(default_angle, dtype=torch.float32, device=device).unsqueeze(0)
    default_angles = default_angles.repeat(envs_num, 1)  # 适配批量（正确）


    joint2motor_idx = [0, 6, 12, 
                       1, 7, 13, 
                       2, 8, 14, 
                       3, 9,      15, 22,
                       4, 10,     16, 23, 
                       5, 11,     17, 24, 
                                  18, 25, 
                                  19, 26, 
                                  20, 27, 
                                  21, 28]

    dof_pos_processed_tran = dof_pos.clone()
    dof_vel_tran = dof_vel.clone()
    for i in range(len(joint2motor_idx)):
            dof_pos_processed_tran[:,i] = dof_pos[:,joint2motor_idx[i]]
            dof_vel_tran[:,i] = dof_vel[:,joint2motor_idx[i]]

    dof_pos_processed_tran = (dof_pos_processed_tran - default_angles)  # 相对偏移计算（正确）
    
    # 3. cmd处理（list→批量tensor，逻辑正确）
    cmd_tensor = torch.tensor(cmd, dtype=torch.float32, device=device).unsqueeze(0)
    cmd_tensor = cmd_tensor.repeat(envs_num, 1)  # 适配批量（正确）

   

    # 4. 拼接顺序（按你要求的顺序，正确）
    states = torch.cat(
        [
            base_ang_vel,               # 3维
            base_gravity_orientation,   # 3维
            cmd_tensor,                 # 3维
            dof_pos_processed_tran,          # 29维
            dof_vel_tran,                    # 29维
            old_action                  # 29维
        ],
        dim=1  # 特征维度拼接（正确）
    )
    
    return states



def simulate_with_opt_dof(handenv, data):
    """
    基于预计算的时序关节数据（data["opt_dof_pos"]）驱动机械手仿真，并收集数据
    Args:
        handenv: 机械手环境实例（包含step方法、关节限制属性、配置参数等）
        data: 字典，需包含键 "opt_dof_pos"，对应时序关节位置数据，形状为 (num_steps, dof_count)
    Returns:
        inputs: 输入数据（关节位置/速度），形状 (num_samples, dof_count * (1+USE_VEL_INPUT))
        targets: 目标数据（指尖位置），形状 (num_samples, num_fingertips * 3)
    """
    # 初始化数据收集列表
    inputs = []
    targets = []
    cfg = handenv.cfg  
    samples_collected = 0

    # --------------------------
    # 1. 提取并校验时序关节数据
    # --------------------------
    # 检查 "opt_dof_pos" 是否在 data 中
    if "opt_dof_pos" not in data:
        raise KeyError("data 字典中缺少键 'opt_dof_pos'，请确保输入数据包含时序关节位置")
    
    opt_dof_pos = data["opt_dof_pos"]
    # 检查数据形状：需为二维数组 (num_steps, dof_count)
    if not isinstance(opt_dof_pos, np.ndarray) or opt_dof_pos.ndim != 2:
        raise ValueError(f"'opt_dof_pos' 需为二维 numpy 数组，当前形状：{opt_dof_pos.shape}")
    
    num_steps, dof_count = opt_dof_pos.shape
    # 检查关节自由度是否与环境匹配
    if dof_count != len(handenv.DOF_LOWER):
        raise ValueError(
            f"关节自由度不匹配：'opt_dof_pos' 每行 {dof_count} 个关节，环境支持 {len(handenv.DOF_LOWER)} 个关节"
        )
    
    print(f"开始基于预计算关节数据仿真，总步数：{num_steps}，目标样本数：{cfg.NUM_SAMPLES}")

    # --------------------------
    # 2. 重置环境并开始仿真
    # --------------------------
    handenv.reset_env()  # 重置机械手到初始状态

    # 遍历每一步预计算的关节数据
    for step_idx in range(num_steps):
        current_joint_pos = opt_dof_pos[step_idx]
        current_joint_pos = np.clip(
            current_joint_pos, 
            handenv.DOF_LOWER, 
            handenv.DOF_UPPER
        )
        # 【关键】保持与原函数一致的动作格式转换（适配 handenv.step() 输入要求）
        action = torch.from_numpy(current_joint_pos).float().to("cuda:0")
        action = action.unsqueeze(0)  # 增加 batch 维度 (1, dof_count)
        action = action.repeat(4, 1)  # 扩展为 (4, dof_count)（根据环境需求调整，原函数逻辑）

        observation, done = handenv.step(action)

       
    # --------------------------
    handenv.close()
    print("仿真结束，环境资源已清理")




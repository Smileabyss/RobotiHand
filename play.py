from envs import HandEnv, HandEnvCPU
from cfg import HandCfg,G1HandCfg,G1Cfg
import numpy as np
from utils import read_pkl, RobotControlGUI
import torch
from train import GRUSeq2Seq
import time

def load_policy_gru(model_path, state_dim, action_dim, H=10, K=10, device="cuda:0"):
    model = GRUSeq2Seq(state_dim, action_dim, hidden=256, K=K).to(device)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model.eval()
    return model



def run_seq2seq_policy_in_env(handenv, policy, H=10, K=10, episode_len=2000, device="cuda:0"):

    state_buffer = []      # 最近 H 个 state
    total_steps = 0        # 计步器

    handenv.reset_env()

    while total_steps < episode_len:

        # ---- 获取当前真实状态 ----
        q, dq, _, _, tip = handenv._read_state()

        q_np = q.cpu().numpy().reshape(-1)
        dq_np = dq.cpu().numpy().reshape(-1)
        tip_np = tip.cpu().numpy().reshape(-1)

        state_np = np.concatenate([q_np, dq_np, tip_np], axis=0)

        # 更新 state history
        state_buffer.append(state_np)
        if len(state_buffer) > H:
            state_buffer.pop(0)

        # 若历史不足 H，执行零动作等待
        if len(state_buffer) < H:
            handenv.step(torch.zeros((1, dq_np.shape[0]), device=device))
            total_steps += 1
            continue

        # ======= 用 GRU 预测未来 K 个动作 =======
        state_tensor = torch.tensor(np.stack(state_buffer), dtype=torch.float32, device=device).unsqueeze(0)


        with torch.no_grad():
            future_actions = policy(state_tensor)   # (1, K, action_dim)

        # ======= 执行未来 K 步动作 =======
        for k in range(K):

            if total_steps >= episode_len:
                break  # rollout 结束

            action_k = future_actions[:, k, :]     # (1, action_dim)

            # 执行动作
            obs, done = handenv.step(action_k)

            total_steps += 1

            # ---- 执行后必须读取真实新状态，放入 buffer ----
            q, dq, _, _, tip = handenv._read_state()
            q_np = q.cpu().numpy().reshape(-1)
            dq_np = dq.cpu().numpy().reshape(-1)
            tip_np = tip.cpu().numpy().reshape(-1)

            next_state_np = np.concatenate([q_np, dq_np, tip_np], axis=0)

            state_buffer.append(next_state_np)
            if len(state_buffer) > H:
                state_buffer.pop(0)

    print("Seq2Seq GRU Policy rollout finished.")


def collect_manipulator_data_with_traj(handenv, target_traj, steps_per_segment=100):
    """
    支持 target_traj (T, env, dof) 输入，插值执行动作，并记录状态。
    """

    target_traj = np.asarray(target_traj)       # (T, env, dof)
    T, n_envs, dof = target_traj.shape

    # 数据缓冲区
    q_list = []
    dq_list = []
    tip_list = []
    action_list = []

    # 重置环境
    handenv.reset_env()

    for k in range(T - 1):
        start = target_traj[k]     # (env, dof)
        end = target_traj[k + 1]   # (env, dof)

        for s in range(steps_per_segment):
            t = s / steps_per_segment
            action_np = start + (end - start) * t  # (env, dof)
            action = torch.from_numpy(action_np).float().to("cuda:0")

            # 执行仿真一步
            obs, done = handenv.step(action)  # 假设 handenv 支持 batch action

            # 读取当前状态
            q, dq, _, _, tip_pos = handenv._read_state()  # 假设 q/dq/tip_pos shape = (env, dof) 或 (env, n_tip, 3)

            # 保存数据
            q_list.append(q.cpu().numpy())
            dq_list.append(dq.cpu().numpy())
            tip_list.append(tip_pos.cpu().numpy())
            action_list.append(action_np)
        # for _ in range(10):
        #     action = torch.from_numpy(end).float().to("cuda:0")
        #     action = action.unsqueeze(0)  # (1, dof)
        #     handenv.step(action)
        #      # 读取当前状态
        #     q, dq,  _, _,tip_pos = handenv._read_state()

        #     # 保存数据
        #     q_list.append(q.cpu().numpy())
        #     dq_list.append(dq.cpu().numpy())
        #     tip_list.append(tip_pos.cpu().numpy())
        #     action_list.append(action_np)

    # ——组装输出——
    data = dict(
        q=np.array(q_list),       # (N, env, dof)
        dq=np.array(dq_list),     # (N, env, dof)
        tip=np.array(tip_list),   # (N, env, n_tip, 3)
        action=np.array(action_list), # (N, env, dof)
    )
    np.save("/home/cyrus/Proprioception_envs/data/collected_data", data)

    # 机器人保持 end target_pos 姿态，不退出仿真
    target_tensor = torch.tensor(target_traj[-1], dtype=torch.float32, device="cuda:0")  # (env, dof)
    while True:
        handenv.step(target_tensor)
        time.sleep(0.01)




if __name__ == "__main__":
 ############ maniptrans play ########################   
    # cfg = G1HandCfg()
    # env = HandEnv(cfg)
    # env.reset_env()
    # gui = RobotControlGUI(env)
    # gui.run()
  ############ expert play ########################   
    # cfg = G1HandCfg()
    # env = HandEnv(cfg)
    # target_angles = np.zeros(38,dtype=float)
    # listidx = [0,1,7,8,15,16,17]
    # listidx_target = [0.311, 0.216,0.718,0.719,0.568,0.143,0.189]
    # for i,idx in enumerate(listidx):
    #     target_angles[idx] = listidx_target[i]
    # target_angles = np.tile(target_angles, (4, 1))   

    # target_angles_1 = target_angles.copy()
    # target_angles_1[:,4] = -0.689
    # target_angles_2 = target_angles_1.copy()
    # target_angles_2[:,0] = -1.458

    # start_angle =  np.zeros(38,dtype=float)
    # start_angle = np.tile(start_angle, (4, 1)) 
    # target_traj = [start_angle,target_angles,target_angles_1,target_angles_2]
    # collect_manipulator_data_with_traj(env,target_traj)
  ############ BC model play ########################   
    cfg = G1HandCfg()
    cfg.envs_num = 1
    env = HandEnv(cfg)
    model_path = "/home/cyrus/Proprioception_envs/model/gru_policy_fail10.pth"
    state_dim = 38 + 38 + 30   # q + dq + tip
    action_dim = 38            # dof
    policy = load_policy_gru(model_path, state_dim, action_dim, H=10, K=10)
    run_seq2seq_policy_in_env(env, policy, H=10, K=10, episode_len=2000)



   

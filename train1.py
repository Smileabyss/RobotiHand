from envs import HandEnv, HandEnvCPU
from cfg import HandCfg,G1HandCfg,G1Cfg
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================
# Dataset
# =====================
class SeqStopDataset(Dataset):
    """
    支持训练 Stop Head：
    - state: q,dq,tip 展平
    - action: 下一步动作
    - stop: 是否是序列最后一步
    """
    def __init__(self, path):
        data = np.load(path, allow_pickle=True).item()

        q = data["q"]       # (T, 1, dof)
        dq = data["dq"]     # (T, 1, dof)
        tip = data["tip"]   # (T, 1, n_tip, 3)
        action = data["action"]  # (T, dof)

        T, B, dof = q.shape  # B=1
        _, _, n_tip, _ = tip.shape

        # 展平 tip
        tip_flat = tip.reshape(T, B, -1)  # (T, B, n_tip*3)

        # state: concat q, dq, tip_flat
        state = np.concatenate([q, dq, tip_flat], axis=2)  # (T, B, state_dim)

        # action_next: pad last step
        action_next = np.zeros((T, B, dof), dtype=np.float32)
        action_next[:-1, :, :] = action[1:, None, :]  # 下一步动作
        action_next[-1, :, :] = action[-1, None, :]  # 最后一步重复

        # stop label: 1 for last step, 0 otherwise
        stop = np.zeros((T, B, 1), dtype=np.float32)
        stop[-1, :, 0] = 1.0

        # flatten batch dim
        self.state = state.reshape(-1, state.shape[2])
        self.action = action_next.reshape(-1, dof)
        self.stop = stop.reshape(-1, 1)

        print("Dataset loaded:", self.state.shape, self.action.shape, self.stop.shape)

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, idx):
        s = torch.tensor(self.state[idx], dtype=torch.float32)
        a = torch.tensor(self.action[idx], dtype=torch.float32)
        st = torch.tensor(self.stop[idx], dtype=torch.float32)
        return s, a, st


# =====================
# GRU + Action Head + Stop Head
# =====================
class GRUStopPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(state_dim, hidden_dim, num_layers, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.stop_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, h=None):
        """
        x: (B, 1, state_dim) 单步输入
        h: hidden state
        """
        out, h_new = self.gru(x, h)  # out: (B, 1, hidden)
        action = self.action_head(out[:, -1, :])  # (B, action_dim)
        stop_prob = torch.sigmoid(self.stop_head(out[:, -1, :]))  # (B,1)
        return action, stop_prob, h_new

# =====================
# 训练函数
# =====================
def train_gru_bc(data_path, epochs=300, batch_size=64, lr=1e-3, device="cuda"):
    dataset = SeqStopDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    state_dim = dataset.state.shape[1]
    action_dim = dataset.action.shape[1]

    model = GRUStopPolicy(state_dim, action_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    print(f"Start Training GRU BC | state_dim={state_dim}, action_dim={action_dim}")
    for ep in range(1, epochs+1):
        total_loss = 0
        for state, action, stop in loader:
            state = state.to(device)
            action = action.to(device)
            stop = stop.to(device)
            
            state = state.unsqueeze(1)  # (B,1,state_dim)
            pred_action, pred_stop, _ = model(state)
            
            loss = mse_loss(pred_action, action) + bce_loss(pred_stop, stop)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * state.size(0)
        print(f"Epoch {ep:03d} | Loss: {total_loss / len(dataset):.6f}")
    
    torch.save(model.state_dict(), "model/gru_stop_policy.pth")
    print("Model saved to model/gru_stop_policy.pth")
    return model

# =====================
# 推理函数
# =====================
def run_gru_policy(handenv, model, max_steps=50, device="cuda"):
    """
    使用 GRU Seq2Seq + Stop Head 模型在仿真中运行策略
    handenv: 仿真环境
    model: GRU 模型，输出 (action, stop_prob, hidden)
    max_steps: 最大 rollout 步数
    device: torch device
    """
    model.eval()
    state_dim = 38 * 2 + 10 * 3

    action_list = []
    stop_list = []
    h = None  # GRU 初始隐藏状态

    handenv.reset_env()

    for step in range(max_steps):
        # ---- 获取当前状态 ----
        q, dq, _, _, tip = handenv._read_state()
        q_np = q.cpu().numpy().reshape(-1)
        dq_np = dq.cpu().numpy().reshape(-1)
        tip_np = tip.cpu().numpy().reshape(-1)
        state_np = np.concatenate([q_np, dq_np, tip_np], axis=0)

        # ---- 构造模型输入 ----
        # (batch=1, seq_len=1, state_dim)
        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)

        # ---- GRU 推理 ----
        with torch.no_grad():
            action, stop_prob, h = model(state_tensor, h)

        action_list.append(action.cpu().numpy())
        stop_list.append(stop_prob.item())

        # ---- 执行动作 ----
        obs, done = handenv.step(action)

        # ---- Stop Head 判定 ----
        if stop_prob.item() > 0.5:
            print(f"Policy terminated at step {step}")
            break

    return np.array(action_list), np.array(stop_list)
# =====================
# 使用示例
# =====================
if __name__ == "__main__":
    data_path = "/home/cyrus/ManipTrans/Proprioception_envs/data/collected_data.npy"
    device = "cuda:0"
    model = train_gru_bc(data_path, epochs=200, batch_size=64, device=device)

    # # 假设 handenv 已经初始化好
    cfg = G1HandCfg()
    env = HandEnv(cfg)
    actions, stops = run_gru_policy(env, model, max_steps=50, device=device)

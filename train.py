import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# =====================
# Dataset
# =====================
class Seq2SeqDataset(Dataset):
    def __init__(self, data_path, H=10, K=10):
        data = np.load(data_path, allow_pickle=True).item()

        q  = data["q"]      # (length, env_num, dof)
        dq = data["dq"]     # (length, env_num, dof)
        tip = data["tip"]   # (length, env_num, n_tip, 3)
        action = data["action"]  # (length, env_num, dof)

        length, env_num, dof = q.shape
        n_tip = tip.shape[2]

        self.H = H
        self.K = K

        # ---- 将每条环境轨迹独立处理，保留轨迹结构 ----
        self.state_seqs = []
        self.action_seqs = []

        for e in range(env_num):
            # 提取单环境轨迹
            q_e = q[:, e, :]         # (length, dof)
            dq_e = dq[:, e, :]       # (length, dof)
            tip_e = tip[:, e, :, :].reshape(length, -1)  # (length, n_tip*3)
            action_e = action[:, e, :]                     # (length, dof)

            # 拼成 state
            state_e = np.concatenate([q_e, dq_e, tip_e], axis=1)  # (length, state_dim)

            # 可取起点数量
            max_start = length - (H + K)
            for idx in range(max_start):
                self.state_seqs.append(state_e[idx : idx + H])         # (H, state_dim)
                self.action_seqs.append(action_e[idx + H : idx + H + K])  # (K, dof)

        self.N = len(self.state_seqs)
        print(f"Dataset loaded: {env_num} environments, total {self.N} samples.")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (
            torch.tensor(self.state_seqs[idx], dtype=torch.float32),
            torch.tensor(self.action_seqs[idx], dtype=torch.float32),
        )

# =====================
# GRU Seq2Seq 模型（不变）
# =====================
class GRUSeq2Seq(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256, K=10):
        super().__init__()
        self.hidden = hidden
        self.K = K
        self.gru = nn.GRU(input_size=state_dim, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, action_dim)

    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(1, B, self.hidden, device=x.device)
        out, _ = self.gru(x, h0)          # (B, H, hidden)
        last = out[:, -1, :]               # (B, hidden)
        pred = self.fc(last).unsqueeze(1).repeat(1, self.K, 1)  # (B, K, action_dim)
        return pred

# =====================
# 训练函数
# =====================
def train_gru_bc(data_path, H=10, K=100, epochs=200, batch_size=32, lr=1e-3, device="cuda"):
    dataset = Seq2SeqDataset(data_path, H, K)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    state_dim = dataset.state_seqs[0].shape[1]
    action_dim = dataset.action_seqs[0].shape[1]

    # model
    model = GRUSeq2Seq(state_dim, action_dim, hidden=256, K=K).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # training
    for ep in range(1, epochs + 1):
        total = 0
        for s, a in loader:
            s, a = s.to(device), a.to(device)
            pred = model(s)               # (B, K, action_dim)
            mse_loss = loss_fn(pred, a)

            # smooth & jerk
            smooth_loss = ((pred[:, 1:, :] - pred[:, :-1, :]) ** 2).mean()
            jerk = pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]
            jerk_loss = (jerk ** 2).mean()

            loss = mse_loss +  smooth_loss + 0.05 * jerk_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item() * s.size(0)

        print(f"Epoch {ep:03d} | Loss: {total / len(dataset):.6f}")

    torch.save(model.state_dict(), "model/gru_policy.pth")
    print("Saved → model/gru_policy.pth")
    return model


# =====================================
# 入口
# =====================================
if __name__ == "__main__":
    train_gru_bc("/home/cyrus/Proprioception_envs/data/collected_data.npy")

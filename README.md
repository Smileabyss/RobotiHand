一个基于 IsaacGym 的 GPU 加速机械臂仿真环境，适用于机器人研究、控制算法开发和操作任务仿真。
特性

    支持 GPU 和 CPU 两种仿真模式，GPU 模式支持多环境并行计算
    实时关节控制与状态监控，包含 38 个关节的位置 / 力混合控制
    交互式 GUI 界面：提供关节滑块控制、数据记录与保存功能（支持 NPZ 和 CSV 格式）
    数据可视化工具：实时显示关节轨迹、位置变化及时间步信息
    支持基于 GRU 的序列到序列（Seq2Seq）控制算法训练与部署
    兼容多种机械臂模型配置（如 23DOF、29DOF 等），可加载 URDF 模型
    包含桌面和物体交互场景，支持抓取等操作任务仿真

前置条件

    Python 3.8+
    CUDA 11.7+（用于 GPU 加速）
    IsaacGym Preview 4

安装步骤
1. 安装 IsaacGym
从官方网站下载 IsaacGym Preview 4，按照文档中的说明完成安装。可通过运行python/examples目录下的joint_monkey.py示例脚本测试安装是否成功。
2. 克隆仓库
bash
运行

git clone https://github.com/Smileabyss/RobotiHand.git
cd RobotiHand

3. 安装依赖
使用 pip
bash
运行

pip install -r requirements.txt

使用 Conda
通过提供的 YAML 文件创建 conda 环境：
bash
运行

conda env create -f environment.yaml
conda activate robothand-env

    注意：环境配置包含 PyTorch 1.13.1+cu117 及其他针对 GPU 加速优化的依赖，可根据系统配置调整版本。

项目结构

    envs/：仿真环境实现
        env_gpu.py：GPU 加速仿真环境，支持多环境并行、关节混合控制（位置 / 力）及物体交互
        env_cpu.py：CPU 基础仿真环境，提供关节信息解析功能
    utils/：工具模块
        robot_control_gui.py：交互式控制界面，支持 38 个关节滑块调节、数据记录与保存
        data_process.py：关节轨迹可视化工具，实时更新位置与轨迹线
        cv2_display.py：基于 OpenCV 的仿真渲染工具
    cfg/：配置文件
        g1handcfg.py：机械臂具体配置参数（如关节限制、仿真步长等）
    train.py/train1.py：GRU 序列到序列控制模型的训练脚本
    play.py：加载预训练模型并在仿真环境中运行的脚本
    assets/：机械臂模型文件（URDF 格式），包含 23DOF、29DOF 等配置

快速开始
1. 基础仿真
python
运行

from envs import HandEnv
from cfg import G1HandCfg

# 初始化配置
cfg = G1HandCfg()
cfg.display = True  # 启用可视化

# 创建环境
env = HandEnv(cfg)

# 运行仿真循环
while True:
    # 示例：零动作（无运动）
    action = torch.zeros(env.dof_count, device=env.DEVICE)
    obs, done = env.step(action)

2. 使用控制 GUI
通过图形界面手动控制关节并记录数据：
python
运行

from envs import HandEnv
from cfg import G1HandCfg
from utils import RobotControlGUI

# 初始化环境
cfg = G1HandCfg()
env = HandEnv(cfg)

# 启动控制界面
gui = RobotControlGUI(env)
gui.run()

3. 训练控制模型
使用收集的数据训练 GRU Seq2Seq 模型：
python
运行

# 从train.py调用训练函数
from train import train_gru_bc

# 训练模型（数据路径需替换为实际收集的数据）
train_gru_bc(
    data_path="/path/to/collected_data.npy",
    epochs=200,
    batch_size=32,
    H=10,  # 历史状态长度
    K=10   # 预测未来动作步数
)

4. 运行预训练模型
加载训练好的模型在环境中执行：
python
运行

from envs import HandEnv
from cfg import G1HandCfg
from play import load_policy_gru, run_seq2seq_policy_in_env

# 初始化环境
cfg = G1HandCfg()
cfg.envs_num = 1
env = HandEnv(cfg)

# 加载模型
model_path = "model/gru_policy.pth"
state_dim = 38 + 38 + 30  # 关节位置 + 关节速度 + 指尖位置
action_dim = 38  # 关节数量
policy = load_policy_gru(model_path, state_dim, action_dim)

# 运行模型
run_seq2seq_policy_in_env(env, policy, episode_len=2000)

关键功能说明

    混合控制模式：支持部分关节位置控制、部分关节力控制，具体可通过配置文件指定
    数据记录：可记录关节位置、速度、指尖位置及动作指令，支持 NPZ（压缩格式）和 CSV（表格格式）保存
    轨迹可视化：实时绘制关节运动轨迹，便于分析运动连续性与平滑性
    多环境并行：GPU 模式下可同时运行多个仿真环境，加速数据收集与算法测试

注意事项

    确保 GPU 支持 CUDA 11.7 及以上版本以获得最佳性能
    可通过配置文件调整compute_device_id和graphics_device_id以匹配 GPU 设置
    机械臂模型关节限制（范围、力限制等）可在assets目录下的 URDF 文件中修改
    遇到物理仿真问题可参考 IsaacGym 官方文档排查

许可证
MIT# RobotiHand
一个基于 IsaacGym 的 GPU 加速机械臂仿真环境，适用于机器人研究、控制算法开发和操作任务仿真。
特性

    支持 GPU 和 CPU 两种仿真模式，GPU 模式支持多环境并行计算
    实时关节控制与状态监控，包含 38 个关节的位置 / 力混合控制
    交互式 GUI 界面：提供关节滑块控制、数据记录与保存功能（支持 NPZ 和 CSV 格式）
    数据可视化工具：实时显示关节轨迹、位置变化及时间步信息
    支持基于 GRU 的序列到序列（Seq2Seq）控制算法训练与部署
    兼容多种机械臂模型配置（如 23DOF、29DOF 等），可加载 URDF 模型
    包含桌面和物体交互场景，支持抓取等操作任务仿真

前置条件

    Python 3.8+
    CUDA 11.7+（用于 GPU 加速）
    IsaacGym Preview 4

安装步骤
1. 安装 IsaacGym
从官方网站下载 IsaacGym Preview 4，按照文档中的说明完成安装。可通过运行python/examples目录下的joint_monkey.py示例脚本测试安装是否成功。
2. 克隆仓库
bash
运行

git clone https://github.com/Smileabyss/RobotiHand.git
cd RobotiHand

3. 安装依赖
使用 pip
bash
运行

pip install -r requirements.txt

使用 Conda
通过提供的 YAML 文件创建 conda 环境：
bash
运行

conda env create -f environment.yaml
conda activate robothand-env

    注意：环境配置包含 PyTorch 1.13.1+cu117 及其他针对 GPU 加速优化的依赖，可根据系统配置调整版本。

项目结构

    envs/：仿真环境实现
        env_gpu.py：GPU 加速仿真环境，支持多环境并行、关节混合控制（位置 / 力）及物体交互
        env_cpu.py：CPU 基础仿真环境，提供关节信息解析功能
    utils/：工具模块
        robot_control_gui.py：交互式控制界面，支持 38 个关节滑块调节、数据记录与保存
        data_process.py：关节轨迹可视化工具，实时更新位置与轨迹线
        cv2_display.py：基于 OpenCV 的仿真渲染工具
    cfg/：配置文件
        g1handcfg.py：机械臂具体配置参数（如关节限制、仿真步长等）
    train.py/train1.py：GRU 序列到序列控制模型的训练脚本
    play.py：加载预训练模型并在仿真环境中运行的脚本
    assets/：机械臂模型文件（URDF 格式），包含 23DOF、29DOF 等配置

快速开始
1. 基础仿真
python
运行

from envs import HandEnv
from cfg import G1HandCfg

# 初始化配置
cfg = G1HandCfg()
cfg.display = True  # 启用可视化

# 创建环境
env = HandEnv(cfg)

# 运行仿真循环
while True:
    # 示例：零动作（无运动）
    action = torch.zeros(env.dof_count, device=env.DEVICE)
    obs, done = env.step(action)

2. 使用控制 GUI
通过图形界面手动控制关节并记录数据：
python
运行

from envs import HandEnv
from cfg import G1HandCfg
from utils import RobotControlGUI

# 初始化环境
cfg = G1HandCfg()
env = HandEnv(cfg)

# 启动控制界面
gui = RobotControlGUI(env)
gui.run()

3. 训练控制模型
使用收集的数据训练 GRU Seq2Seq 模型：
python
运行

# 从train.py调用训练函数
from train import train_gru_bc

# 训练模型（数据路径需替换为实际收集的数据）
train_gru_bc(
    data_path="/path/to/collected_data.npy",
    epochs=200,
    batch_size=32,
    H=10,  # 历史状态长度
    K=10   # 预测未来动作步数
)

4. 运行预训练模型
加载训练好的模型在环境中执行：
python
运行

from envs import HandEnv
from cfg import G1HandCfg
from play import load_policy_gru, run_seq2seq_policy_in_env

# 初始化环境
cfg = G1HandCfg()
cfg.envs_num = 1
env = HandEnv(cfg)

# 加载模型
model_path = "model/gru_policy.pth"
state_dim = 38 + 38 + 30  # 关节位置 + 关节速度 + 指尖位置
action_dim = 38  # 关节数量
policy = load_policy_gru(model_path, state_dim, action_dim)

# 运行模型
run_seq2seq_policy_in_env(env, policy, episode_len=2000)

关键功能说明

    混合控制模式：支持部分关节位置控制、部分关节力控制，具体可通过配置文件指定
    数据记录：可记录关节位置、速度、指尖位置及动作指令，支持 NPZ（压缩格式）和 CSV（表格格式）保存
    轨迹可视化：实时绘制关节运动轨迹，便于分析运动连续性与平滑性
    多环境并行：GPU 模式下可同时运行多个仿真环境，加速数据收集与算法测试

注意事项

    确保 GPU 支持 CUDA 11.7 及以上版本以获得最佳性能
    可通过配置文件调整compute_device_id和graphics_device_id以匹配 GPU 设置
    机械臂模型关节限制（范围、力限制等）可在assets目录下的 URDF 文件中修改
    遇到物理仿真问题可参考 IsaacGym 官方文档排查

许可证
MIT
Smileabyss/RobotiHand
environment.yml

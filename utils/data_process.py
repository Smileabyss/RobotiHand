import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def read_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    # 打印数据类型和关键信息
    print("数据类型：", type(data))
    print("数据 keys：", data.keys())
    for key in data.keys():
        print(f"{key}:",data[key].shape)

    return data



def visualize_joint_trajectory(data, interval=50):

    joint_data = data
    # 验证数据形状
    if joint_data.ndim != 3 or joint_data.shape[1] != 18 or joint_data.shape[2] != 3:
        raise ValueError(f"关节数据形状应为 [length, 18, 3]，实际为 {joint_data.shape}")

    # 初始化3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴范围
    x_min, x_max = np.min(joint_data[..., 0]), np.max(joint_data[..., 0])
    y_min, y_max = np.min(joint_data[..., 1]), np.max(joint_data[..., 1])
    z_min, z_max = np.min(joint_data[..., 2]), np.max(joint_data[..., 2])
    ax.set_xlim(x_min * 1.1, x_max * 1.1)
    ax.set_ylim(y_min * 1.1, y_max * 1.1)
    ax.set_zlim(z_min * 1.1, z_max * 1.1)

    ax.set_xlabel('X Axis (m)')
    ax.set_ylabel('Y Axis (m)')
    ax.set_zlabel('Z Axis (m)')
    ax.set_title('18-Joint Trajectory Visualization')

    # --------------------------
    # 关键修复：兼容地创建颜色列表
    # --------------------------
    # 1. 获取颜色映射对象
    try:
        # Matplotlib 3.6+ 推荐方式
        cmap = plt.colormaps.get('rainbow')
    except AttributeError:
        # 旧版本兼容方式
        cmap = plt.get_cmap('rainbow')

    # 2. 生成18个均匀分布的颜色
    num_joints = 18
    colors = cmap(np.linspace(0, 1, num_joints))  # 生成一个RGBA颜色数组

    # 初始化18个关节点（不预先设置颜色）
    scatter = ax.scatter([], [], [], s=50)

    # 初始化轨迹线
    trajectories = []
    for i in range(num_joints):
        # 为每条轨迹线指定颜色
        line, = ax.plot([], [], [], '--', color=colors[i], alpha=0.6)
        trajectories.append(line)

    # 初始化时间文本
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)

    # 初始化函数
    def init():
        scatter._offsets3d = ([], [], [])
        for line in trajectories:
            line.set_data([], [])
            line.set_3d_properties([])
        time_text.set_text('')
        return scatter, *trajectories, time_text

    # 更新函数
    def update(frame):
        current_joints = joint_data[frame]

        x = current_joints[:, 0]
        y = current_joints[:, 1]
        z = current_joints[:, 2]

        # 更新关节点位置
        scatter._offsets3d = (x, y, z)
        # 更新关节点颜色
        scatter.set_facecolors(colors)

        # 更新轨迹线
        for i in range(num_joints):
            traj_x = joint_data[:frame + 1, i, 0]
            traj_y = joint_data[:frame + 1, i, 1]
            traj_z = joint_data[:frame + 1, i, 2]
            trajectories[i].set_data(traj_x, traj_y)
            trajectories[i].set_3d_properties(traj_z)

        # 更新时间文本
        time_text.set_text(f'Time Step: {frame}/{len(joint_data) - 1}')

        return scatter, *trajectories, time_text

    # 创建动画
    ani = animation.FuncAnimation(
        fig, update, frames=len(joint_data), init_func=init,
        interval=interval, blit=False, repeat=True
    )

    # 显示动画
    plt.tight_layout()
    plt.show()

    return ani

# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":

    
    path = "/home/cyrus/ManipTrans/Proprioception_envs/data/expert_data/left_hand@2.pkl"
    data = read_pkl(path)
    data = data['opt_joints_pos']
    
    # 调用可视化函数
    ani = visualize_joint_trajectory(data, interval=30)
    
    # 可选：保存动画（需要安装ffmpeg）
    # writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('joint_trajectory.mp4', writer=writer)

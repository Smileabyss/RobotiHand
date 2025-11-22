import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import torch
import time
from datetime import datetime
import os

class RobotControlGUI:
    def __init__(self, handenv):
        self.handenv = handenv
        self.root = tk.Tk()
        self.root.title("机械臂实时控制与数据记录")
        self.root.geometry("1200x800")
        
        # 关节状态
        self.joint_values = np.zeros(38, dtype=np.float32)  # 38个关节
        self.joint_scales = []
        self.joint_lower_limits = handenv.DOF_LOWER # 假设是一个包含38个关节下限的numpy数组
        self.joint_upper_limits = handenv.DOF_UPPER # 假设是一个包含38个关节上限的numpy数组
        
        # 数据记录
        self.recorded_data = []
        self.is_recording = False
        self.record_start_time = None
        
        # 初始化界面
        self._create_widgets()
        
        # 启动仿真循环
        self.update_simulation()
        
    def _create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="机械臂关节控制", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 关节控制框架（左右两栏）
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 创建滑块（左栏：0-18关节）
        for i in range(19):
            self._create_joint_slider(left_frame, i, i)
        
        # 创建滑块（右栏：19-37关节）
        for i in range(19, 38):
            self._create_joint_slider(right_frame, i, i-19)
        
        # 控制按钮框架
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # 记录按钮
        self.record_button = ttk.Button(control_frame, text="开始记录", command=self.toggle_recording)
        self.record_button.grid(row=0, column=0, padx=10)
        
        # 保存按钮
        save_button = ttk.Button(control_frame, text="保存数据", command=self.save_data)
        save_button.grid(row=0, column=1, padx=10)
        
        # 重置按钮
        reset_button = ttk.Button(control_frame, text="重置关节", command=self.reset_joints)
        reset_button.grid(row=0, column=2, padx=10)
        
        # 状态显示
        self.status_label = ttk.Label(control_frame, text="状态: 就绪", font=("Arial", 12))
        self.status_label.grid(row=0, column=3, padx=20)
        
        # 数据统计
        self.data_count_label = ttk.Label(control_frame, text="已记录样本数: 0", font=("Arial", 12))
        self.data_count_label.grid(row=0, column=4, padx=20)
        
    def _create_joint_slider(self, parent, joint_idx, row):
        # 关节框架
        joint_frame = ttk.Frame(parent, padding="2")
        joint_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # 关节标签
        joint_label = ttk.Label(joint_frame, text=f"关节{joint_idx:02d}", width=6)
        joint_label.grid(row=0, column=0, sticky=tk.W)
        
        # 滑块
        # scale = ttk.Scale(joint_frame, from_=0.0, to=1.5, orient=tk.HORIZONTAL,
        #                  command=lambda val, idx=joint_idx: self.update_joint_value(idx, float(val)))
        # scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        # scale.set(0.0)
        # 根据关节索引获取该关节的具体限制
        lower_limit = self.joint_lower_limits[joint_idx]
        upper_limit = self.joint_upper_limits[joint_idx]
        
        # 使用关节自身的限制来创建滑动条
        scale = ttk.Scale(joint_frame, from_=lower_limit, to=upper_limit, orient=tk.HORIZONTAL,  length=200,
                        command=lambda val, idx=joint_idx: self.update_joint_value(idx, float(val)))
        scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        scale.set(0.0)
        # 数值显示
        value_var = tk.StringVar()
        value_label = ttk.Label(joint_frame, textvariable=value_var, width=8)
        value_label.grid(row=0, column=2, sticky=tk.E)
        
        self.joint_scales.append((scale, value_var))
        
    def update_joint_value(self, joint_idx, value):
        # 更新关节值
        self.joint_values[joint_idx] = value
        
        # 更新显示
        scale, var = self.joint_scales[joint_idx]
        var.set(f"{value:.3f}")
        
    def toggle_recording(self):
        if not self.is_recording:
            # 开始记录
            self.is_recording = True
            self.record_start_time = time.time()
            self.record_button.config(text="停止记录")
            self.status_label.config(text="状态: 正在记录")
        else:
            # 停止记录
            self.is_recording = False
            self.record_button.config(text="开始记录")
            self.status_label.config(text="状态: 记录已停止")
        
    def reset_joints(self):
        # 重置所有关节到0
        self.joint_values = np.zeros(38, dtype=np.float32)
        for i in range(38):
            self.joint_scales[i][0].set(0.0)
            self.joint_scales[i][1].set("0.000")
        
    def update_simulation(self):
        # 构造动作
        action = torch.from_numpy(self.joint_values).float().to(self.handenv.DEVICE)
        action = action.unsqueeze(0).repeat(4, 1)  # 适配多环境
        
        # 执行仿真步骤
        observation, done = self.handenv.step(action)
        
        # 记录数据
        if self.is_recording:
            # 获取当前状态
            q, dq, fingertip_pos, _, _ = self.handenv._read_state()
            
            # 记录数据（时间戳、关节位置、关节速度、指尖位置）
            timestamp = time.time() - self.record_start_time
            self.recorded_data.append({
                "timestamp": timestamp,
                "joint_pos": q.cpu().numpy(),
                "joint_vel": dq.cpu().numpy(),
                "fingertip_pos": fingertip_pos.cpu().numpy(),
                "action": action.cpu().numpy()

            })
            
            # 更新统计
            self.data_count_label.config(text=f"已记录样本数: {len(self.recorded_data)}")
        
        # 定期更新
        self.root.after(50, self.update_simulation)  # 50ms更新一次
        
    def save_data(self):
        if not self.recorded_data:
            self.status_label.config(text="状态: 没有可保存的数据")
            return
        
        # 选择保存路径
        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy数组", "*.npz"), ("CSV文件", "*.csv"), ("所有文件", "*.*")],
            title="保存记录数据"
        )
        
        if not filepath:
            return
        
        # 保存为NumPy数组
        if filepath.endswith(".npz"):
            timestamps = np.array([d["timestamp"] for d in self.recorded_data])
            joint_pos = np.array([d["joint_pos"] for d in self.recorded_data])
            joint_vel = np.array([d["joint_vel"] for d in self.recorded_data])
            fingertip_pos = np.array([d["fingertip_pos"] for d in self.recorded_data])
            
            np.savez_compressed(
                filepath,
                timestamps=timestamps,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                fingertip_pos=fingertip_pos,
                recorded_at=datetime.now().isoformat()
            )
            
            self.status_label.config(text=f"状态: 数据已保存到 {os.path.basename(filepath)}")
        
        # 保存为CSV
        elif filepath.endswith(".csv"):
            with open(filepath, "w") as f:
                # 写入表头
                headers = ["timestamp"] + [f"joint_pos_{i}" for i in range(38)] + \
                          [f"joint_vel_{i}" for i in range(38)] + [f"fingertip_pos_{i}" for i in range(12)]
                f.write(",".join(headers) + "\n")
                
                # 写入数据
                for data in self.recorded_data:
                    row = [str(data["timestamp"])] + \
                          list(data["joint_pos"].flatten()) + \
                          list(data["joint_vel"].flatten()) + \
                          list(data["fingertip_pos"].flatten())
                    f.write(",".join(map(str, row)) + "\n")
            
            self.status_label.config(text=f"状态: 数据已保存到 {os.path.basename(filepath)}")
        
    def run(self):
        self.root.mainloop()


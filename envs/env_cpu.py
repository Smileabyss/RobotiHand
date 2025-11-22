import os
import numpy as np
import isaacgym.gymapi as gymapi
from isaacgym import gymtorch
from cfg import G1HandCfg
import torch
import sys
import time
from os.path import join
import torchvision


# ----------------- 手部仿真环境类 -----------------
class HandEnvCPU:
    def __init__(self, cfg: G1HandCfg):
        self.cfg = cfg
        self.gym = None
        self.sim = None
        self.env = None
        self.hand_actor = None
        self.viewer = None
        self.asset = None
        self.camera = None
        
        # 核心状态变量
        self.dof_count = 0
        self.dof_names = []
        self.fingertip_rb_indices = []  # 指尖刚体索引
        self.DOF_LOWER = None
        self.DOF_UPPER = None
        self.DOF_VELOCITY_LIMIT = None
        self.DOF_EFFORT_LIMIT = None
        self.force_control_indices = []  # 力控制关节的索引
        self.pos_control_indices = []    # 位置控制关节的索引

        # 初始化流程
        self._init_sim()          # 创建仿真环境
        self._load_asset()   # 加载手部URDF
        self._init_actor()   # 创建手部Actor
        self._init_viewer()       # 创建可视化窗口
        self._prepare_sim()       # 预热仿真
 
        
    def _init_sim(self):
        """创建Isaac Gym仿真"""
        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.SIM_DT
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = False
        sim_params.physx.use_gpu = False
        sim_params.up_axis = gymapi.UP_AXIS_Z  # Z轴向上
        
        self.sim = self.gym.create_sim(-1, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("创建仿真实例失败")
        
        # 添加地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # 地面法向量（Z轴向上）
        self.gym.add_ground(self.sim, plane_params)

    def _load_asset(self):
        # 原有逻辑不变，加载URDF并获取关节名称
        options = gymapi.AssetOptions()
        options.fix_base_link = self.cfg.fix_base_link
        options.collapse_fixed_joints = self.cfg.collapse_fixed_joints
        options.disable_gravity = self.cfg.disable_gravity
        options.use_mesh_materials = self.cfg.use_mesh_materials
        
        root_path = os.path.dirname(self.cfg.URDF_PATH)
        file_name = os.path.basename(self.cfg.URDF_PATH)
        self.asset = self.gym.load_urdf(self.sim, root_path, file_name, options)
        
        if self.asset is None:
            raise RuntimeError(f"URDF加载失败: {self.cfg.URDF_PATH}")
        
        # 获取关节信息（名称和数量）
        self.dof_count, self.dof_names = self._get_joint_dof_info()
        # 查找指尖刚体索引
        self.fingertip_rb_indices = self._find_fingertip_links()
        
        # 新增：根据配置的关节名称，映射到控制模式的索引
        self._map_control_indices()  # 关键：建立名称→索引的映射
        
        print(f"成功加载手部模型：{self.cfg.URDF_PATH}")
        print(f"关节数量：{self.dof_count}")
        print(f"力控制关节（索引）：{self.force_control_indices} → 名称：{[self.dof_names[i] for i in self.force_control_indices]}\n")
        print(f"位置控制关节（索引）：{self.pos_control_indices} → 名称：{[self.dof_names[i] for i in self.pos_control_indices]}\n")
        print(f"找到指尖刚体索引：{self.fingertip_rb_indices} → 名称：{[self.gym.get_asset_rigid_body_name(self.asset, i) for i in self.fingertip_rb_indices]}")
    
    
    def _init_actor(self):
        """创建手部Actor并配置关节属性"""
        # 创建环境（原有逻辑）
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.env = self.gym.create_env(self.sim, lower, upper, 1)
       
        
        # 手部姿态（原有逻辑）
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.5, 0.5)
        pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
        self.hand_actor = self.gym.create_actor(
            self.env, self.asset, pose, "hand", group=0, filter=0
        )
        
        # 配置关节属性
        dof_props = self.gym.get_asset_dof_properties(self.asset)
        
        # 1. 配置力控制关节（DOF_MODE_EFFORT）
        for idx in self.force_control_indices:
            dof_props['driveMode'][idx] = gymapi.DOF_MODE_EFFORT  
        
        # 2. 配置位置控制关节（DOF_MODE_POS）
        for idx in self.pos_control_indices:
            dof_props['driveMode'][idx] = gymapi.DOF_MODE_POS  
            dof_props['stiffness'][idx] = self.cfg.stiffness[idx]
            dof_props['damping'][idx] = self.cfg.damping[idx]
        
        # 应用关节属性
        self.gym.set_actor_dof_properties(self.env, self.hand_actor, dof_props)
        
        # 读取关节限位
        dof_props = self.gym.get_actor_dof_properties(self.env, self.hand_actor)
        self.DOF_LOWER = np.array(dof_props['lower'], dtype=np.float32)  
        self.DOF_UPPER = np.array(dof_props['upper'], dtype=np.float32)  
        self.DOF_VELOCITY_LIMIT = np.array(dof_props['velocity'], dtype=np.float32)  
        self.DOF_EFFORT_LIMIT = np.array(dof_props['effort'], dtype=np.float32) 

    def _init_viewer(self):
        """创建可视化窗口"""
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise RuntimeError("创建可视化窗口失败")
        
        # 相机视角（正对手部原生方向）
        cam_pos = gymapi.Vec3(1.2, 0.5, 0.8)
        cam_target = gymapi.Vec3(0.0, 0.5, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, self.env, cam_pos, cam_target)

    def _prepare_sim(self):
        """仿真预热"""
        self.gym.prepare_sim(self.sim)
        print("仿真预热中...")
        for _ in range(150):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        print("预热完成，开始正常仿真")

    def _map_control_indices(self):
        """将配置中的关节名称映射为索引"""
        # 遍历所有关节名称，匹配力控制/位置控制的配置
        for idx, name in enumerate(self.dof_names):
            if name in self.cfg.force_control_dof_names:
                self.force_control_indices.append(idx)
            elif name in self.cfg.pos_control_dof_names:
                self.pos_control_indices.append(idx)
            else:
                self.pos_control_indices.append(idx)
                print(f"警告：关节 {name} 未在控制配置中，默认使用位置控制")
        
        # 检查是否有重复配置
        overlap = set(self.force_control_indices) & set(self.pos_control_indices)
        if overlap:
            raise ValueError(f"关节索引 {overlap} 同时出现在力控制和位置控制列表中，请检查配置")
        
    def _find_fingertip_links(self):
        """查找指尖刚体索引"""
        found = []
        rb_count = self.gym.get_asset_rigid_body_count(self.asset)
        for i in range(rb_count):
            rb_name = self.gym.get_asset_rigid_body_name(self.asset, i)
            for ft_name in self.cfg.fingertip_names:
                if ft_name.lower() in rb_name.lower():
                    found.append(i)
                    break
        return list(dict.fromkeys(found))  # 去重
    
    def _get_joint_dof_info(self):
        """获取关节数量和名称"""
        dof_count = self.gym.get_asset_dof_count(self.asset)
        names = []
        try:
            for i in range(dof_count):
                names.append(self.gym.get_asset_dof_name(self.asset, i))
        except Exception:
            names = [f"dof_{i}" for i in range(dof_count)]
        return dof_count, names
    
    
    def _read_state(self):
        """读取手部状态"""
        dof_states = self.gym.get_actor_dof_states(self.env, self.hand_actor, gymapi.STATE_ALL)
        dof_pos = np.zeros(self.dof_count, dtype=np.float32)
        dof_vel = np.zeros(self.dof_count, dtype=np.float32)
        for i in range(self.dof_count):
            dof_pos[i] = dof_states[i]['pos']
            dof_vel[i] = dof_states[i]['vel']
        
        # 读取指尖位置
        body_states = self.gym.get_actor_rigid_body_states(self.env, self.hand_actor, gymapi.STATE_ALL)
        fingertip_pos = np.zeros((len(self.fingertip_rb_indices), 3), dtype=np.float32)
        for i in range(len(self.fingertip_rb_indices)):
            fingertip_pos[i] = body_states['pose'][self.fingertip_rb_indices[i]]['p'].view((np.float32, 3))
        
        return dof_pos, dof_vel, fingertip_pos
    
    def step(self, action):
        """
        单步仿真：接收一个包含所有关节指令的列表，自动区分位置控制和力控制
        Args:
            action: 所有关节的控制指令，形状 (dof_count,) 
                    - 位置控制关节：元素为目标位置（单位：rad）
                    - 力控制关节：元素为力矩指令（单位：N·m）
        Returns:
            next_dof_pos: 所有关节的下一时刻位置，(dof_count,)
            next_dof_vel: 所有关节的下一时刻速度，(dof_count,)
            fingertip_pos: 指尖位置，(num_fingertips, 3)
            done: 是否结束（恒为False）
        """
        # 校验action长度（必须等于总关节数）
        if len(action) != self.dof_count:
            raise ValueError(f"action长度错误：预期 {self.dof_count}（总关节数），实际 {len(action)}")
        action = np.array(action, dtype=np.float32)  
        

        # 应用位置控制指令（裁剪到关节限位）
        if len(self.pos_control_indices) != 0:
            pos_action = action[self.pos_control_indices]  
            full_pos_target = self._read_state()[0].copy()  # 用当前位置初始化所有关节目标
            for i, idx in enumerate(self.pos_control_indices):
                full_pos_target[idx] = np.clip(pos_action[i], self.DOF_LOWER[idx], self.DOF_UPPER[idx])
            self.gym.set_actor_dof_position_targets(self.env, self.hand_actor, full_pos_target)

        # 应用力控制指令（裁剪到力矩限制）
        if len(self.force_control_indices) != 0:
            force_action = action[self.force_control_indices]  
            full_force = np.zeros(self.dof_count, dtype=np.float32)  # 初始化所有关节力为0
            for i, idx in enumerate(self.force_control_indices):
                full_force[idx] = np.clip(force_action[i], -self.DOF_EFFORT_LIMIT[idx], self.DOF_EFFORT_LIMIT[idx])
            self.gym.apply_actor_dof_efforts(self.env, self.hand_actor, full_force)

        # 5. 执行仿真并更新可视化
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        # 6. 读取下一状态
        next_dof_pos, next_dof_vel, fingertip_pos = self._read_state()

        return next_dof_pos, next_dof_vel, fingertip_pos, False

    
    def close(self):
        """关闭环境，释放资源"""
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if self.env is not None:
            self.gym.destroy_env(self.env)
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
        print("环境已关闭，资源释放完成")


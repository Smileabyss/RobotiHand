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
from utils import Cv2Display



class HandEnv:
    def __init__(self, cfg: G1HandCfg):
        self.cfg = cfg
        self.gym = None
        self.sim = None
        self.env = None
        self.hand_actor = None
        self.viewer = None
        self.asset = None
        self.camera = None
        self.camera_handlers = [] if (self.cfg.display or self.cfg.record) else None
        self.camera_obs = [] if (self.cfg.display or self.cfg.record) else None
        
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
        self.DEVICE = torch.device(f"cuda:{self.cfg.compute_device_id}")
      
        

        # 初始化流程
        self._init_sim()          # 创建仿真环境
        self._load_asset()   # 加载手部URDF
        self._init_actor_env()   # 创建手部Actor and env
        self._init_viewer()       # 创建可视化窗口
        self._prepare_sim()       # 预热仿真
    
        self._set_renderers(self.cfg.display)
        self.set_camera()
        

    def _set_renderers(self, display):
        self._rgb_viewr_renderer = Cv2Display("IsaacGym") if display else None
        
    def _init_sim(self):
        """创建Isaac Gym仿真（GPU版本）"""
        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = self.cfg.SIM_DT
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.8
        compute_device = self.cfg.compute_device_id 
        graphics_device = self.cfg.graphics_device_id 

        # 启动 GPU 版本模拟器 
        self.sim = self.gym.create_sim(
            compute_device,        # 物理仿真 GPU
            graphics_device,       # 图形渲染 GPU
            gymapi.SIM_PHYSX,
            sim_params
        )

        if self.sim is None:
            raise RuntimeError("创建仿真实例失败（请检查GPU是否可用）")
        
        # 添加地面
        self.creat_plane()

        print(f"GPU 仿真已创建：compute_device={compute_device}, graphics_device={graphics_device}")

    def creat_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction =1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def _load_asset(self):
        # 原有逻辑不变，加载URDF并获取关节名称
        options = gymapi.AssetOptions()
        options.fix_base_link = self.cfg.fix_base_link
        options.collapse_fixed_joints = self.cfg.collapse_fixed_joints
        options.disable_gravity = self.cfg.disable_gravity
        options.use_mesh_materials = self.cfg.use_mesh_materials
        options.flip_visual_attachments = self.cfg.flip_visual_attachments
        options.thickness = 0.001
        options.angular_damping = 20
        options.linear_damping = 20
        options.max_linear_velocity = 50
        options.max_angular_velocity = 100

        
        root_path = os.path.dirname(self.cfg.URDF_PATH)
        file_name = os.path.basename(self.cfg.URDF_PATH)
        self.asset = self.gym.load_urdf(self.sim, root_path, file_name, options)
        if self.asset is None:
            raise RuntimeError(f"URDF加载失败: {self.cfg.URDF_PATH}")
        
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.asset)
        for element in rigid_shape_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(self.asset, rigid_shape_props_asset)

        
        # 获取关节信息（名称和数量）
        self.dof_count, self.dof_names = self._get_joint_dof_info()
        # 查找指尖刚体索引
        self.fingertip_rb_indices = self._find_fingertip_links()
        # 新增：根据配置的关节名称，映射到控制模式的索引
        self._map_control_indices()  # 关键：建立名称→索引的映射
        
        print(f"成功加载模型：{self.cfg.URDF_PATH}")
        print(f"关节数量：{self.dof_count}")
        print(f"力控制关节（索引）：{self.force_control_indices} → 名称：{[self.dof_names[i] for i in self.force_control_indices]}\n")
        print(f"位置控制关节（索引）：{self.pos_control_indices} → 名称：{[self.dof_names[i] for i in self.pos_control_indices]}\n")
        print(f"找到指尖刚体索引：{self.fingertip_rb_indices} → 名称：{[self.gym.get_asset_rigid_body_name(self.asset, i) for i in self.fingertip_rb_indices]}")
    
    def _init_actor_env(self):
        """批量创建环境+机械臂Actor（补充张量解析必需的维度信息）"""
        # 1. 多环境配置
        self.num_envs = self.cfg.envs_num  # 并行环境数量
        self.env_spacing = self.cfg.env_spacing  # 环境间距
        
        # 计算环境网格布局
        self.num_envs_rows = int(torch.sqrt(torch.tensor(self.num_envs)).item())
        self.num_envs_cols = self.num_envs // self.num_envs_rows

        self.envs = []  
        self.actors = []  # 仅存储机械臂Actor
        self.env_origins = []
        self.camera_handlers = []
        self._global_indices = []

        # 关键新增1：每个环境的总Actor数（桌面+方块+机械臂）
        self.per_env_actor_num = 3  # 固定为3，与创建顺序一致
        # 关键新增2：机械臂在当前环境中的Actor索引（创建顺序：桌面(0)→方块(1)→机械臂(2)）
        self.robot_actor_idx_in_env = 2
        self.cube_actor_idx_in_env = 1
        self.table_actor_idx_in_env = 0
        # 关键新增3：获取机械臂DOF数量和刚体数量（用于后续解析）
        self.num_dofs = self.gym.get_asset_dof_count(self.asset)
        self.robot_body_num = self.gym.get_asset_rigid_body_count(self.asset)  # 机械臂自身的刚体数

        # 在循环外预先创建桌面和方块的Asset，避免重复创建
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, 0.6, 1, 0.05, table_asset_options)
        self.table_body_num = self.gym.get_asset_rigid_body_count(self.table_asset)  # 桌面刚体数（固定为1）

        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.fix_base_link = False
        self.cube_asset = self.gym.create_box(self.sim, 0.04, 0.04, 0.04, cube_asset_options)
        self.cube_body_num = self.gym.get_asset_rigid_body_count(self.cube_asset)  # 方块刚体数（固定为1）

        # 关键新增4：每个环境的总刚体数（桌面+方块+机械臂）
        self.per_env_total_body_num = self.table_body_num + self.cube_body_num + self.robot_body_num
        
        # 2. 循环批量创建环境、机械臂Actor、桌面和方块
        for env_idx in range(self.num_envs):
            # 2.1 计算当前环境的位置
            col = env_idx % self.num_envs_cols
            row = env_idx // self.num_envs_cols
            env_origin = gymapi.Vec3(
                col * self.env_spacing,
                row * self.env_spacing,
                0.0
            )
            self.env_origins.append(env_origin)  # 保存原点，用于重置

            # 创建单个环境
            env = self.gym.create_env(
                self.sim,
                env_origin - gymapi.Vec3(self.env_spacing/2, self.env_spacing/2, 0.0),
                env_origin + gymapi.Vec3(self.env_spacing/2, self.env_spacing/2, self.env_spacing),
                1
            )

            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env,
                        isaac_gym=self.gym,
                    )
                )

            # 创建桌面Actor（顺序1：索引0）
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(env_origin.x, env_origin.y, 0.7)
            table_handle = self.gym.create_actor(
                env, self.table_asset, table_pose, "table", group=0, filter=0
            )
            self.gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.2, 0.1))

            # 创建方块Actor（顺序2：索引1）
            cube_pose = gymapi.Transform()
            cube_pose.p = gymapi.Vec3(env_origin.x, env_origin.y, 0.775)
            cube_handle =  self.gym.create_actor(
                env, self.cube_asset, cube_pose, "cube", group=0, filter=0
            )
             # set  color to be dark gray
            self.gym.set_rigid_body_color(env, cube_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))
            props = self.gym.get_actor_rigid_body_properties(env, cube_handle)
            props[0].mass = 0.0625  # 根据密度计算的质量，例如松木方块
            self.gym.set_actor_rigid_body_properties(env, cube_handle, props, recomputeInertia=True)

            # 2.2 创建机械臂Actor（顺序3：索引2，核心）
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(env_origin.x - 0.4, env_origin.y, 0.8)  # 避开桌面
            pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
            robot_handle = self.gym.create_actor(
                env, self.asset, pose, "hand", group=0, filter=0
            )
            self.actors.append(robot_handle)

            # 2.3 配置关节属性（启用力传感器，避免之前的警告）
            dof_props = self.gym.get_asset_dof_properties(self.asset)
            for idx in self.force_control_indices:
                dof_props['driveMode'][idx] = gymapi.DOF_MODE_EFFORT
                dof_props["damping"][idx] = 0.0
                dof_props['stiffness'][idx] = 0.0
            for idx in self.pos_control_indices:
                dof_props['driveMode'][idx] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][idx] = self.cfg.stiffness[idx]
                dof_props['damping'][idx] = self.cfg.damping[idx]
            self.gym.set_actor_dof_properties(env, robot_handle, dof_props)

            self.envs.append(env)

    #########[[0,1,2][3,4,5],,,]
        for env_idx in range(self.cfg.envs_num):
            start_idx = env_idx * self.per_env_actor_num
            table_handle = start_idx + 0
            cube_handle  = start_idx + 1
            robot_handle = start_idx + 2
            self._global_indices.append([table_handle, cube_handle, robot_handle])
        self._global_indices = torch.tensor(self._global_indices, dtype=torch.int32, device=self.DEVICE)

        # self.gym.refresh_actor_root_state_tensor(self.sim)
        self.initial_full_root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.initial_full_root_states = gymtorch.wrap_tensor(self.initial_full_root_states)

        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.actors[0])
        self.DOF_LOWER = np.array(dof_props['lower'], dtype=np.float32)
        self.DOF_UPPER = np.array(dof_props['upper'], dtype=np.float32)
        self.DOF_VELOCITY_LIMIT = np.array(dof_props['velocity'], dtype=np.float32)
        self.DOF_EFFORT_LIMIT = np.array(dof_props['effort'], dtype=np.float32)
    def reset_env(self):
        num_envs = self.num_envs  
        num_dofs = self.dof_count  # 机械臂关节数（如38）
        default_angle = np.array(self.cfg.default_angle, dtype=np.float32) 
        default_cube_pos = np.array(self.cfg.default_cube_pos, dtype=np.float32)
        
        # -------------------------- 1. 构造 DOF 状态张量（机械臂关节状态）--------------------------
        dof_state_shape = (num_envs * num_dofs, 2)
        self._dof_state = torch.zeros(dof_state_shape, dtype=torch.float32, device=self.DEVICE)
        for env_idx in range(num_envs):
            start_idx = env_idx * num_dofs
            end_idx = start_idx + num_dofs
            self._dof_state[start_idx:end_idx, 0] = torch.tensor(default_angle, device=self.DEVICE)  # 关节位置
            self._dof_state[start_idx:end_idx, 1] = 0.0  # 关节速度
        
        # -------------------------- 2. 构造根状态张量（所有Actor的根状态）--------------------------
        total_actors = num_envs * self.per_env_actor_num
        new_root_states = torch.zeros((total_actors, 13), dtype=torch.float32, device=self.DEVICE)
        default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.DEVICE)
        
        for env_idx in range(num_envs):
            env_start_row = env_idx * self.per_env_actor_num
            env_origin = self.env_origins[env_idx]
            env_origin_tensor = torch.tensor([env_origin.x, env_origin.y, env_origin.z], device=self.DEVICE)
            
            # 桌面（Actor 0）
            table_row = env_start_row + 0
            table_pos = env_origin_tensor + torch.tensor([0.0, 0.0, 0.7], device=self.DEVICE)
            new_root_states[table_row, 0:3] = table_pos
            new_root_states[table_row, 3:7] = default_quat
            
            # 物块（Actor 1）
            cube_row = env_start_row + 1
            cube_pos = table_pos + torch.tensor(default_cube_pos, device=self.DEVICE)
            new_root_states[cube_row, 0:3] = cube_pos
            new_root_states[cube_row, 3:7] = default_quat
            
            # 机械臂（Actor 2）
            robot_row = env_start_row + 2
            robot_pos = env_origin_tensor + torch.tensor([-0.4, 0.0, 0.8], device=self.DEVICE)
            new_root_states[robot_row, 0:3] = robot_pos
            new_root_states[robot_row, 3:7] = torch.tensor([0, 0, 0.0, 1.0], device=self.DEVICE)

        # -------------------------- 3. 构造位置控制目标张量（机械臂关节目标）--------------------------
        self._pos_control = torch.zeros((num_envs * num_dofs,), dtype=torch.float32, device=self.DEVICE)
        for env_idx in range(num_envs):
            start_idx = env_idx * num_dofs
            end_idx = start_idx + num_dofs
            self._pos_control[start_idx:end_idx] = torch.tensor(default_angle, device=self.DEVICE)
        
        # -------------------------- 4. 关键：构造正确的 actor_indices（全局索引列表）--------------------------
        robot_global_indices = self._global_indices[:, 2].flatten()  # 形状：(num_envs,) = (4,)
        all_actors_global_indices = self._global_indices.flatten()  # 形状：(total_actors,) = (12,)
        
        # -------------------------- 5. 调用 API--------------------------
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state.contiguous()),
            gymtorch.unwrap_tensor(robot_global_indices.contiguous()),  
            len(robot_global_indices),
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(new_root_states.contiguous()),
            gymtorch.unwrap_tensor(all_actors_global_indices.contiguous()), 
            len(all_actors_global_indices),
        )

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control.contiguous()),
            gymtorch.unwrap_tensor(robot_global_indices.contiguous()), 
            len(robot_global_indices),
        )

        # -------------------------- 6. 刷新状态并返回 --------------------------
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self._refresh()
        dof_pos, dof_vel, base_ang_vel, base_gravity_orientation, _ = self._read_state()  
        self.observation = torch.cat([base_ang_vel, base_gravity_orientation, dof_pos, dof_vel], dim=1)
        return self.observation

        

   

    def _init_viewer(self):
        if self.cfg.headless:
            self.viewer = None
            return

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise RuntimeError("创建可视化窗口失败")
        
        self.enable_viewer_sync = True

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")  # ESC退出
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")  # V切换同步
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "record_frames")  # R录制帧
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "resetenv") 

        # 相机设置
        cam_pos = gymapi.Vec3(1.2, 0.5, 0.8)    # 相机位置
        cam_target = gymapi.Vec3(0.0, 0.5, 0.5) # 瞄准点
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)  

    def set_camera(self):
        if self.camera_obs is not None:
            for env, handle in zip(self.envs, self.camera_handlers):
                self.camera_obs.append(
                    gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(self.sim, env, handle, gymapi.IMAGE_COLOR)
                    )
                )
    def render(self, mode="rgb_array"):
        self.last_frame_time = time.time()
        if self._rgb_viewr_renderer is not None:
            rgbs = torch.stack(self.camera_obs)[..., :-1]  # RGBA -> RGB
            rgbs = rgbs.permute(0, 3, 1, 2)  # (n, 3, H, W)
            N = rgbs.shape[0]
            rgb_to_display = torchvision.utils.make_grid(rgbs, nrow=N // 2)
            self._rgb_viewr_renderer(rgb_to_display)

        if self.viewer is not None:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames
                elif evt.action == "resetenv" and evt.value > 0:
                    self.reset_env()

            # fetch results
            if self.DEVICE != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

                # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # this code will slow down the rendering to real time
                now = time.time()
                delta = now - self.last_frame_time
                if self.cfg.render_fps  < 0:
                    # render at control frequency
                    render_dt = self.cfg.SIM_DT * self.control_freq_inv  # render every control step
                else:
                    render_dt = 1.0 / self.cfg.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.cfg.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(
                    self.viewer,
                    join(self.record_frames_dir, f"frame_{self.control_steps}.png"),
                )



    def create_camera(
        self,
        *,
        env,
        isaac_gym,
        ):
        """
        Only create front camera for view purpose
        """
        if self.cfg.record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera
    
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
                # 未配置的关节默认使用位置控制（或根据需求改为报错）
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
    
        
    @staticmethod
    def quat_rotate_vector(quat, vec):
        """用四元数旋转向量（适配1维输入：quat shape=(4,), vec shape=(3,)）"""
        # 1维四元数拆分（直接按索引取标量，无需按dim拆分）
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        # 1维向量拆分
        vec_x, vec_y, vec_z = vec[0], vec[1], vec[2]
        
        # 旋转计算（数学公式不变，操作标量）
        rotated_x = (
            qw*qw*vec_x 
            + 2*qy*qw*vec_z 
            - 2*qz*qw*vec_y 
            + qx*qx*vec_x 
            + 2*qy*qx*vec_y 
            + 2*qz*qx*vec_z 
            - qz*qz*vec_x 
            - qy*qy*vec_x
        )
        rotated_y = (
            2*qx*qy*vec_x 
            + qy*qy*vec_y 
            + 2*qz*qw*vec_x 
            + qw*qw*vec_y 
            - 2*qx*qw*vec_z 
            + 2*qz*qy*vec_z 
            - qz*qz*vec_y 
            - qx*qx*vec_y
        )
        rotated_z = (
            2*qx*qz*vec_x 
            + 2*qy*qz*vec_y 
            + qz*qz*vec_z 
            - qw*qw*vec_z 
            - 2*qx*qw*vec_y 
            - qx*qx*vec_z 
            - qy*qy*vec_z 
            + 2*qy*qw*vec_x
        )
        
        # 返回1维向量（shape=(3,)）
        return torch.tensor([rotated_x, rotated_y, rotated_z], dtype=quat.dtype, device=quat.device)

    @staticmethod
    def get_gravity_projection(quat):
        """计算重力投影（适配1维四元数：quat shape=(4,)）"""
        # 世界坐标系重力向量（1维，shape=(3,)）
        gravity_world = torch.tensor([0.0, 0.0, -1.0], dtype=quat.dtype, device=quat.device)
        
        # 调用旋转函数（输入均为1维）
        gravity_proj = HandEnv.quat_rotate_vector(quat, gravity_world)
        
        # 返回1维重力投影（shape=(3,)）
        return gravity_proj
 
    def _update_states(self):
        """使用 Tensor API 读取状态（已修正接触力计算）"""
        # 1. 解析Actor根状态张量（所有Actor：num_envs * 3 个）
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, self.per_env_actor_num, 13)
        self._robot_root_state = self._root_state[:, self.robot_actor_idx_in_env, :]

        # 2. 解析DOF状态张量（仅机械臂有DOF）
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, self.num_dofs, 2)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]

        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        # _dof_force = self.gym.acquire_dof_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        # self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)

        # 4. 机械臂基态相关
        self.base_ang_vel = self._robot_root_state[:, 10:13]
        self.base_gravity_orientation = torch.zeros((self.cfg.envs_num, 3), dtype=torch.float32, device=self.DEVICE)
        for env_idx in range(self.cfg.envs_num):
            self.base_gravity_orientation[env_idx] = self.get_gravity_projection(self._robot_root_state[env_idx, 3:7])

        # 5. 解析刚体状态张量（所有刚体）
        self.body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.body_state_tensor = gymtorch.wrap_tensor(self.body_state_tensor)
        
        robot_body_offset = self.table_body_num + self.cube_body_num
        self.fingertip_pos = torch.zeros((self.cfg.envs_num, len(self.fingertip_rb_indices), 3), dtype=torch.float32, device=self.DEVICE)
        
        for env_idx in range(self.cfg.envs_num):
            for i, rb_idx in enumerate(self.fingertip_rb_indices): 
                global_idx = env_idx * self.per_env_total_body_num + robot_body_offset + rb_idx
                self.fingertip_pos[env_idx, i] = self.body_state_tensor[global_idx, 0:3]
    # def _update_states(self):
    #     """使用 Tensor API 读取状态"""
    #     _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
    #     _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
    #     _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
    #     # _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

    #     self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
    #     self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
    #     self._q = self._dof_state[..., 0]
    #     self._qd = self._dof_state[..., 1]
    #     self._base_state = self._root_state[:, 0, :]
    #     self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
    #     # self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)

    #     self.base_ang_vel = torch.zeros((self.cfg.envs_num, 3), dtype=torch.float32, device=self.DEVICE)
    #     self.base_gravity_orientation = torch.zeros((self.cfg.envs_num, 3), dtype=torch.float32, device=self.DEVICE)
    #     for env_idx in range(self.cfg.envs_num):
    #         self.base_ang_vel[env_idx] = self._root_state[env_idx, 0, 10:13]
    #         self.base_gravity_orientation[env_idx] = self.get_gravity_projection(self._root_state[env_idx, 0, 3:7])


    #     # 读取并拆分刚体状态
    #     self.body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
    #     self.body_state_tensor = gymtorch.wrap_tensor(self.body_state_tensor)
    #     single_env_body_num = self.gym.get_actor_rigid_body_count(self.envs[0], self.actors[0])
    #     self.fingertip_pos = torch.zeros((self.cfg.envs_num, len(self.fingertip_rb_indices), 3), dtype=torch.float32, device=self.DEVICE)
    #     for env_idx in range(self.cfg.envs_num):
    #         for i, idx in enumerate(self.fingertip_rb_indices): 
    #             global_idx = idx + env_idx*single_env_body_num
    #             self.fingertip_pos[env_idx, i] = self.body_state_tensor[global_idx, 0:3]


    
    def step(self, action):
        """
        单步仿真总控：整合多步物理+相机渲染
        """
        self.pre_physics_step(action)

        # 多步物理仿真：控制频率与物理频率的比值
        # 配置项含义：self.cfg.control_freq_inv = 物理步数/控制步数（默认1）
        self.control_freq_inv = self.cfg.control_freq_inv  

        for i in range(self.control_freq_inv):
            if self._rgb_viewr_renderer is not None or self.viewer is not None:
                self.render()
            self.gym.simulate(self.sim)

        if self.camera_obs is not None:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

        if self.camera_obs is not None:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        self.post_physics_step()

        if self.camera_obs is not None:
            self.gym.end_access_image_tensors(self.sim)


        # 返回结果
        done = False
        return  self.observation, done



    def pre_physics_step(self, action):
        """
        仿真前预处理：校验action合法性、拆分控制指令、应用位置/力控制（原逻辑迁移）
        Args:
            action: 输入的原始控制指令
        """
        # 1. 校验action类型
        if not isinstance(action, torch.Tensor):
            action = torch.from_numpy(action).float().to(self.DEVICE)
        else:
            action = action.float().to(self.DEVICE)
        

        # 2. 刷新当前状态
        self._refresh()

        # 3. 应用位置控制指令
        if len(self.pos_control_indices) != 0:
            pos_action = action[:, self.pos_control_indices]  
            current_dof_pos, _, _,_,_ = self._read_state()
            full_pos_target = current_dof_pos.clone() 
            for env_idx in range(self.num_envs):  
                for i, dof_idx in enumerate(self.pos_control_indices): 
                    full_pos_target[env_idx, dof_idx] = torch.clamp(
                        pos_action[env_idx, i],
                        min=self.DOF_LOWER[dof_idx],
                        max=self.DOF_UPPER[dof_idx]
                    )
            
            #「扁平化张量」
            # (4, 38) → (152,)，按环境顺序拼接，API会自动分环境应用
            full_pos_target_flat = full_pos_target.flatten()
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(full_pos_target_flat))

        # 4. 应用力控制指令
        if len(self.force_control_indices) != 0:
            force_action = action[:, self.force_control_indices]
            full_force = torch.zeros((self.num_envs, self.dof_count), dtype=torch.float32, device=self.DEVICE)
            for env_idx in range(self.num_envs): 
                for i, dof_idx in enumerate(self.force_control_indices):  
                    full_force[env_idx, dof_idx] = torch.clamp(
                        force_action[env_idx, i],
                        min=-self.DOF_EFFORT_LIMIT[dof_idx],
                        max=self.DOF_EFFORT_LIMIT[dof_idx]
                    )
            
            full_force_flat = full_force.flatten()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(full_force_flat))

    def post_physics_step(self):
        """
        仿真后后处理：刷新仿真后状态、读取下一时刻数据
        """
        # 1. 刷新仿真后的最新状态
        self._refresh()

        # 2. computer observation
        dof_pos, dof_vel, base_ang_vel, base_gravity_orientation, _ = self._read_state()  
        
        self.observation = torch.cat(
            [
                base_ang_vel,                
                base_gravity_orientation,   
                dof_pos,                    
                dof_vel                   
            ],
            dim=1  
        )

    
    def _refresh(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def _read_state(self):
        return self._q, self._qd, self.base_ang_vel, self.base_gravity_orientation, self.fingertip_pos
    
    def close(self):
        """关闭环境，释放资源"""
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if self.env is not None:
            self.gym.destroy_env(self.env)
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
        print("环境已关闭，资源释放完成")


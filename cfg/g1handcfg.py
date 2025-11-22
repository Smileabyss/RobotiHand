import torch
import os
import time
from .handcfg import HandCfg
class G1HandCfg:
    def __init__(self):
        # 数据路径
        self.pkl_path = "/home/cyrus/data/retargeting/OakInk-v2/mano2rh56f1_lh/scene_01__A003++seq__01857017aef944761b78__2023-04-21-19-28-11@12.pkl"
        self.URDF_PATH =  "/home/cyrus/Proprioception_envs/assets/g1_f1/RH56F1.urdf"
        self.ASSET_ROOT = os.path.dirname(self.URDF_PATH)
        self.handcfg = HandCfg()
        
        # 仿真参数
        self.SIM_DT = 1.0 / 120.0
        self.USE_VEL_INPUT = False
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stiffness = [100,100,50,50,40,40,40] + self.handcfg.stiffness + [100,100,50,50,40,40,40] + self.handcfg.stiffness
        self.damping = [2,2,2,2,2,2,2] + self.handcfg.damping + [2,2,2,2,2,2,2] + self.handcfg.damping
                    
        self.fix_base_link = True
        self.collapse_fixed_joints = False
        self.disable_gravity = False
        self.use_mesh_materials = True
        self.flip_visual_attachments = False
        self.compute_device_id = 0      # 用于物理和张量运算
        self.graphics_device_id = 0     # 用于渲染


        # 渲染相关默认参数
        self.headless = False
        self.display = False
        self.record = False
        self.render_fps = -1  # -1 表示按控制频率渲染
        self.last_frame_time = time.time()
        self.record_frames = False
        self.record_frames_dir = "/home/cyrus/Proprioception_envs/data/recorded_frames"
        self.enable_viewer_sync = True
        self.control_steps = 0  # 控制步数（用于录制文件名）
        self.camera_obs = None  # 相机图像缓存（后续由相机API赋值）
        self.control_freq_inv = 1

        # 环境
        self.envs_num = 1 
        self.env_spacing = 2.0  

        # 手部运动参数
        self.STEPS_UP = 200    # 0°→90°步数
        self.STEPS_DOWN = 200  # 90°→0°步数
        self.NUM_SAMPLES = 60000
        
        # 名称List
        self.default_cube_pos = [-0.1,0.15,0.05]
        self.default_angle = [0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0]
        self.fingertip_names = ["tip", "finger_tip"]
        self.force_control_dof_names = [] #[
        #     'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
        #     'left_elbow_joint',
        #     'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',

        #     'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 
        #     'right_elbow_joint', 
        #     'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'

        # ]
        self.pos_control_dof_names = []#[
        #     'left_index_1_joint', 'left_index_2_joint',
        #     'left_little_1_joint', 'left_little_2_joint', 
        #     'left_middle_1_joint', 'left_middle_2_joint', 
        #     'left_ring_1_joint', 'left_ring_2_joint', 
        #     'left_thumb_1_joint', 'left_thumb_2_joint', 'left_thumb_3_joint', 'left_thumb_4_joint',  

        #     'right_index_1_joint', 'right_index_2_joint', 
        #     'right_little_1_joint', 'right_little_2_joint', 
        #     'right_middle_1_joint', 'right_middle_2_joint', 
        #     'right_ring_1_joint', 'right_ring_2_joint', 
        #     'right_thumb_1_joint', 'right_thumb_2_joint', 'right_thumb_3_joint', 'right_thumb_4_joint'
        # ]
         


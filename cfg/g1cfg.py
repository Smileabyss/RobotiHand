import torch
import os
import time
class G1Cfg:
    def __init__(self):
        # 数据路径
        self.pkl_path = "/home/cyrus/data/retargeting/OakInk-v2/mano2rh56f1_lh/scene_01__A003++seq__01857017aef944761b78__2023-04-21-19-28-11@12.pkl"
        self.URDF_PATH =  "/home/cyrus/Proprioception_envs/assets/g1/g1_29dof_v1.urdf"
        self.ASSET_ROOT = os.path.dirname(self.URDF_PATH)
        
        # 仿真参数
        self.SIM_DT = 0.003
        self.USE_VEL_INPUT = False
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
                    
        self.fix_base_link = False
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
        self.control_freq_inv = 2

        # 环境
        self.envs_num = 1 
        self.env_spacing = 2.0  

        # 手部运动参数
        self.STEPS_UP = 200    # 0°→90°步数
        self.STEPS_DOWN = 200  # 90°→0°步数
        self.NUM_SAMPLES = 60000
        
        # 名称List
        self.fingertip_names = ["tip", "finger_tip"]
        self.dof_names =  [
              'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
              'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
              'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
              'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
              'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
              'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 
              'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        

        self.stiffness = [200,150,150,200,20,20,
                          200,150,150,200,20,20,
                          200,200,200,
                          100,100,50,50,
                          40,40,40,
                          100,100,50,50,
                          40,40,40]
        self.damping = [5,5,5,5,2,2,
                        5,5,5,5,2,2,
                        5,5,5,
                        2,2,2,2,
                        2,2,2,
                        2,2,2,2,
                        2,2,2]
        self.kps = [200, 200, 200,
            150, 150, 200,
            150, 150, 200,
            200, 200,      100, 100,
            20,  20,       100, 100,
            20,  20,       50,  50, 
                        50,  50,
                        40,  40,
                        40,  40,
                        40,  40]

        self.kds = [5, 5, 5, 
            5, 5, 5, 
            5, 5, 5, 
            5, 5,     2, 2,
            2, 2,     2, 2, 
            2, 2,     2, 2,
                  2, 2,
                  2, 2,
                  2, 2,
                  2, 2]

        self.default_angle= [-0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
                         -0.2, 0.0, 0.0, 0.42, -0.23, 0.0,
                         0.0, 0.0, 0.0,
                         0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
                         0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0]   
        
        self.default_angles = [-0.2,  -0.2,  0.0,
                  0.0,   0.0,  0.0,
                  0.0,   0.0,  0.0, 
                  0.42,  0.42,       0.35,  0.35,
                 -0.23, -0.23,       0.18, -0.18, 
                  0.0,   0.0,        0.0,   0.0, 
                                     0.87,  0.87, 
                                     0.0,   0.0, 
                                     0.0,   0.0, 
                                     0.0,   0.0]

        self.body_names = ['pelvis', 
                'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
                'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
                'waist_yaw_link', 'waist_roll_link', 'torso_link',
                'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 
                'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
                'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 
                'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']
        
        self.force_control_dof_names =  ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
              'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
              'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
              'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
              'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
              'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 
              'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'] 
        
        self.pos_control_dof_names = []
         


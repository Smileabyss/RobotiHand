# 前置条件
Python 3.8+
CUDA 11.7+（用于 GPU 加速）
IsaacGym Preview 4

# ENV
conda create -y -n maniptrans python=3.8
conda activate maniptrans
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117


# Install
1. 安装 IsaacGym
从官方网站下载 IsaacGym Preview 4，按照文档中的说明完成安装。可通过运行python/examples目录下的joint_monkey.py示例脚本测试安装是否成功。

3. 克隆仓库
git clone https://github.com/Smileabyss/RobotiHand.git
cd RobotiHand

4. 安装依赖
   
使用 pip运行
pip install -r requirements.txt

or 使用 Conda通过提供的 YAML 文件创建 conda 环境：

conda env create -f environment.yaml
conda activate robothand-env

 

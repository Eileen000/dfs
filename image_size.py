from PIL import Image
import sys

# 获取命令行传入的图像文件路径
image_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-imagen/liaoxiaoshan/imagen_backup/tmp/Diff_llie-dfs/dicm1/input.png'

# 打开图像文件
with Image.open(image_path) as img:
    # 获取图像的尺寸
    width, height = img.size

# 输出图像尺寸
print(f"Width: {width}, Height: {height}")

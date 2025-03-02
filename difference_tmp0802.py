import cv2
import numpy as np

def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_images(image_path1, image_path2, gamma, gamma_corrected_output_path, subtracted_output_path):
    # 读取图片
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # 检查图片是否读取成功
    if img1 is None:
        print(f"错误：无法读取 {image_path1}")
        return
    if img2 is None:
        print(f"错误：无法读取 {image_path2}")
        return
    
    # 确认两张图片具有相同大小和通道数
    if img1.shape != img2.shape:
        print("错误：两张图片的大小或通道数不一致")
        return

    # 对低光图像进行 gamma 校正
    img2_gamma_corrected = gamma_correction(img2, gamma)
    
    # 保存 gamma 校正后的图像
    cv2.imwrite(gamma_corrected_output_path, img2_gamma_corrected)
    # print(f"gamma 校正后的结果已保存到 {gamma_corrected_output_path}")

    # 图片相减
    subtracted_image = cv2.subtract(img1, img2_gamma_corrected)
    
    # 保存相减结果
    cv2.imwrite(subtracted_output_path, subtracted_image)
    # print(f"相减后的结果已保存到 {subtracted_output_path}")

# 输入图片路径
image_path1 = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-imagen/liaoxiaoshan/imagen_backup/tmp/Diff_llie-dfs/outputs/psld-samples-llie/test_t600_g10.0_o1.0_a10.0_e0.5_s0.3/results/00708.png'
image_path2 = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-imagen/liaoxiaoshan/imagen_backup/tmp/Diff_llie-dfs/outputs/psld-samples-llie/test_t600_g10.0_o1.0_a10.0_e0.5_s0.3/measurements/L_00708.png'
gamma_corrected_output_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-imagen/liaoxiaoshan/imagen_backup/tmp/Diff_llie-dfs/outputs/psld-samples-llie/test_t600_g10.0_o1.0_a10.0_e0.5_s0.3/gamma_corrected_image.png'
subtracted_output_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-imagen/liaoxiaoshan/imagen_backup/tmp/Diff_llie-dfs/outputs/psld-samples-llie/test_t600_g10.0_o1.0_a10.0_e0.5_s0.3/subtracted_image.png'
gamma_value = 1.9  # 可以根据你的低光图像调整gamma值

process_images(image_path1, image_path2, gamma_value, gamma_corrected_output_path, subtracted_output_path)

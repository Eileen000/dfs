import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import pywt
from PIL import Image

def denoise_fft(image_data):
    denoised_image = np.zeros_like(image_data)
    for channel in range(image_data.shape[2]):
        single_channel = image_data[:, :, channel]
        
        # 对图像进行傅立叶变换
        f_transform = fft2(single_channel)
        f_transform_shifted = fftshift(f_transform)

        # 创建滤波器
        rows, cols = single_channel.shape
        crow, ccol = rows // 2, cols // 2
        keep_fraction = 0.1
        size = int(min(rows, cols) * keep_fraction)
        mask = np.zeros((rows, cols))
        mask[crow-size:crow+size, ccol-size:ccol+size] = 1

        # 应用滤波器
        f_transform_shifted = f_transform_shifted * mask

        # 逆傅立叶变换
        f_transform = ifftshift(f_transform_shifted)
        single_channel_denoised = ifft2(f_transform)
        single_channel_denoised = np.abs(single_channel_denoised)
        
        denoised_image[:, :, channel] = single_channel_denoised
        
    return denoised_image

def denoise_wavelet(image_data):
    denoised_image = np.zeros_like(image_data)
    for channel in range(image_data.shape[2]):
        single_channel = image_data[:, :, channel]
        
        # 对图像进行小波变换
        coeffs = pywt.wavedec2(single_channel, 'db1', level=2)  # 使用 Daubechies 小波 db1

        # 应用阈值去噪
        threshold = 30  # 设置阈值
        coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') if isinstance(c, np.ndarray) else c for c in coeffs]
        
        # 逆小波变换
        single_channel_denoised = pywt.waverec2(coeffs_denoised, 'db1')
        
        denoised_image[:, :, channel] = single_channel_denoised
        
    return denoised_image

def save_image(image, path):
    # 将图像数据缩放至 0-255 且转换为uint8
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    # 保存图像
    Image.fromarray(image).save(path)

# 路径
img_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-imagen/liaoxiaoshan/imagen_backup/tmp/Diff_llie-dfs/outputs/psld-samples-llie/_invH1_t600_g20.0_o10.0_a0.0_e0.4_s0.3_gb0.0_ic0.0_re_0.0_cd_0.0/measurements/L_00690.png'
 
# 读取图片并转为 RGB 图像
pic = Image.open(img_path).convert('RGB')

#大小设定，w 和 h 参数根据实际需求调整
w, h = 512, 512  
origin_img = np.array(pic.resize((w, h), resample=Image.LANCZOS)).astype(np.float32) / 255.0

# 用傅立叶变换去噪
image_denoised_fft = denoise_fft(origin_img)
save_image(image_denoised_fft, 'denoised_fft.png')

# 用小波变换去噪
image_denoised_wavelet = denoise_wavelet(origin_img)
save_image(image_denoised_wavelet, 'denoised_wavelet.png')

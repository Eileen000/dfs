import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

def bilateral_filter(image_path, output_path, sigma_color=0.1, sigma_space=10):
    # 读取图片
    image = Image.open(image_path)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # 转换为 [1, 3, H, W]

    # 定义双边滤波层
    class BilateralFilter(nn.Module):
        def __init__(self, sigma_color, sigma_space):
            super(BilateralFilter, self).__init__()
            self.sigma_color = sigma_color
            self.sigma_space = sigma_space

        def forward(self, x):
            # 使用PyTorch的卷积层实现双边滤波
            # 先生成高斯权重
            channels = x.size(1)
            spatial_sigma = torch.tensor(self.sigma_space, dtype=torch.float32)
            color_sigma = torch.tensor(self.sigma_color, dtype=torch.float32)

            spatial_weight = self._generate_gaussian_kernel(spatial_sigma)
            color_weight = self._generate_gaussian_kernel(color_sigma)

            # 应用高斯权重
            filtered = x.clone()
            for i in range(channels):
                filtered[:, i, :, :] = F.conv2d(
                    x[:, i, :, :].unsqueeze(1),
                    color_weight.expand(channels, 1, -1, -1),
                    padding=self._calculate_padding(spatial_sigma),
                )
            filtered = F.conv2d(
                filtered,
                spatial_weight.expand(channels, 1, -1, -1),
                padding=self._calculate_padding(spatial_sigma),
            )

            return filtered

        def _generate_gaussian_kernel(self, sigma):
            # 生成二维高斯核
            dim = int(2 * np.ceil(sigma) + 1)
            x = torch.arange(-dim // 2 + 1, dim // 2 + 1)
            kernel = torch.exp(-x**2 / (2 * sigma**2))
            return kernel[:, None] * kernel[None, :]

        def _calculate_padding(self, sigma):
            # 计算填充大小
            return int(2 * np.ceil(sigma) - 1) // 2

    # 应用双边滤波
    bilateral_filter_layer = BilateralFilter(sigma_color, sigma_space)
    filtered_image = bilateral_filter_layer(image)

    # 保存图片
    filtered_image = filtered_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    filtered_image = filtered_image.astype(np.uint8)
    output_image = Image.fromarray(filtered_image)
    output_image.save(output_path)


# 示例路径，实际使用时需要替换为具体的文件路径
input_image_path = '/home/eileen/Diffusion/DuduZhu/Diff_llie/outputs/psld-samples-llie/107/results/23.png'
output_image_path = '/home/eileen/Diffusion/DuduZhu/Diff_llie/tmp/bilateral.png'

img = cv2.imread(input_image_path)
bilateral_filter_img1 = cv2.bilateralFilter(img, 9, 150, 150)

cv2.imwrite(output_image_path, img)
# 调用双边滤波函数
# bilateral_filter(input_image_path, output_image_path)

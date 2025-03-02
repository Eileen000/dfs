import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from measurements import get_noise, DarkenOperator

img_path = os.path.join('/home/eileen/Diffusion/DuduZhu/Diff_llie/lol_samples/493.png')
torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Prepare Operator and noise
operator = DarkenOperator(
    device=torch_device, w=256, h=256,
    img_path=img_path, dtype=torch.float32, beta=1,
)

img = operator.origin_img
if img.ndim == 2:
    img = img[:, :, None]

L_img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous().unsqueeze(0).to(device=torch_device,
                                                                                dtype=torch.float32)
E0_img = operator.get_illum(1).to(device=torch_device, dtype=torch.float32)
gamma = 2-torch.exp(E0_img)-0.7
# print(2-torch.exp(E0_img))
E0_high_img = E0_img**(gamma)
I0_img = L_img / E0_img * E0_high_img

vutils.save_image(L_img, '/home/eileen/Diffusion/DuduZhu/Diff_llie/tmp/L.png')
vutils.save_image(L_img / E0_img, '/home/eileen/Diffusion/DuduZhu/Diff_llie/tmp/I.png')
vutils.save_image(E0_img, '/home/eileen/Diffusion/DuduZhu/Diff_llie/tmp/E0.png')
vutils.save_image(E0_high_img, '/home/eileen/Diffusion/DuduZhu/Diff_llie/tmp/gammaE0.png')
vutils.save_image(I0_img, '/home/eileen/Diffusion/DuduZhu/Diff_llie/tmp/I0.png')

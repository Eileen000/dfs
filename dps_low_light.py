import argparse
import os
import time
from typing import Optional, Union
from typing import Tuple

import cv2
import torch
import torch.nn as nn
import torchvision.utils as vutils
from diffusers import (
    DDPMScheduler,
    DiffusionPipeline,
    ImagePipelineOutput,
)
from diffusers import UNet2DModel
from diffusers.utils import (
    randn_tensor,
)
from pytorch_lightning import seed_everything

from measurements import get_noise, DarkenOperator


# Code source: https://github.com/huggingface/diffusers/blob/main/examples/community/dps_pipeline.py


def img_save(image, path):
    image = image.clamp(0, 1)
    image = (image.detach().cpu().permute(0, 2, 3, 1)).squeeze(0).numpy()
    images = (image * 255).round().astype("uint8")
    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, images)


# import torch.nn.functional as F


class L2Loss(nn.Module):
    def forward(self, input, target):
        return torch.linalg.norm(input - target)
        # return torch.sqrt(F.mse_loss(input, target, reduction='mean')) * 255.
        # return torch.sqrt(F.mse_loss(input, target, reduction='sum'))
        # return torch.sqrt(torch.sum((input - target) ** 2))


class DPSPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        # self.loss_fn = L2Loss(reduction='mean')
        self.loss_fn = L2Loss()

    @torch.no_grad()
    def __call__(
            self,
            measurement: torch.Tensor,
            operator,
            batch_size: Optional[int] = 1,
            generator: Optional[torch.Generator] = None,
            num_inference_steps: Optional[int] = 1000,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            step_size: Optional[float] = 10.0,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            save_path: Optional[str] = None,
            print_freq: Optional[int] = 10,
            noiser=None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
        image_dtype = measurement.dtype

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=image_dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            with torch.enable_grad():
                # 1. predict noise model_output
                image = image.detach().requires_grad_()
                model_output = self.unet(image, t).sample

                # 2. compute previous image x'_{t-1} and original prediction x0_{t}
                scheduler_out = self.scheduler.step(model_output, t, image, generator=generator)
                image_pred, origi_pred = scheduler_out.prev_sample, scheduler_out.pred_original_sample

                # 3. compute y'_t = f(x0_{t})
                measurement_pred = operator.forward(origi_pred, is_latent=True)
                # measurement_pred = noiser(measurement_pred)

                # 4. compute loss = d(y, y'_t-1)
                loss = self.loss_fn(measurement, measurement_pred)
                loss.backward()

                if i % print_freq == 0:
                    # print(f"[{i:04d}:{t:04d}] loss: {loss.item():.6f}")
                    name, ext = os.path.splitext(save_path)
                    img_save((origi_pred + 1.0) / 2.0, f"{name}_x0_pred_{t.item():04d}{ext}")

                with torch.no_grad():
                    image_pred = image_pred - step_size * image.grad
                    image = image_pred.detach()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir to read samples from",
        default="samples/"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/psld-samples-llie"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=999,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce. A.k.a. batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="/home/ljx/Diffusion/ddpm-celebahq-256", #"google/ddpm-celebahq-256",
        help="model id",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=10.0,
        help="reconstruction error",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.6,
        help="beta for lime",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="lime step",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="number of ddim sampling steps",
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weight_dtype = torch.float32

    model = UNet2DModel.from_pretrained(opt.model_id).to(device=torch_device, dtype=weight_dtype)
    model.enable_xformers_memory_efficient_attention()

    scheduler = DDPMScheduler.from_pretrained(opt.model_id)
    scheduler.set_timesteps(opt.ddim_steps)

    pipeline = DPSPipeline(
        unet=model,
        scheduler=scheduler,
    )
    pipeline.to(torch_device=torch_device, torch_dtype=weight_dtype)

    batch_size = opt.n_samples

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'measurements'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(opt.outdir, 'inter'), exist_ok=True)

    imgsName = sorted(os.listdir(opt.indir))

    for i, imgName in enumerate(imgsName):

        #########################################################
        # set up operator and measurement
        #########################################################
        img_path = os.path.join(opt.indir, imgName)
        save_path = os.path.join(opt.outdir, 'inter', f'inter_{imgName}')
        print('>>>>>>>>> ', img_path, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # Prepare Operator and noise
        operator = DarkenOperator(
            device=torch_device, w=opt.W, h=opt.H,
            img_path=img_path, dtype=torch.float32, beta=opt.beta,
        )
        noiser = get_noise(name='gaussian', sigma=0.01)

        img = operator.origin_img
        if img.ndim == 2:
            img = img[:, :, None]

        L_img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous().unsqueeze(0).to(device=torch_device,
                                                                                        dtype=weight_dtype)
        E0_img = operator.get_illum(opt.step).to(device=torch_device, dtype=weight_dtype)

        vutils.save_image(L_img, os.path.join(opt.outdir, 'measurements', f'L_{imgName}'))
        vutils.save_image(E0_img, os.path.join(opt.outdir, 'measurements', f'E0_{imgName}'))
        vutils.save_image(L_img * E0_img ** -0.6, os.path.join(opt.outdir, 'measurements', f'I0_{imgName}'))

        L_img = L_img * 2.0 - 1.0
        #########################################################

        image = pipeline(
            measurement=L_img,
            operator=operator,
            height=opt.H,
            width=opt.W,
            num_inference_steps=opt.ddim_steps,
            output_type="pil",
            save_path=save_path,
            print_freq=opt.print_freq,
            noiser=noiser,
            step_size=opt.step_size,
            return_dict=True,
        ).images[0]

        image.save(os.path.join(opt.outdir, 'results', f'I_{imgName}'))

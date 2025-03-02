import argparse
import os
import time
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import yaml
from PIL import Image
from diffusers import (
    ImagePipelineOutput,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipelineOutput,
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.schedulers import RePaintScheduler
from diffusers.schedulers import DDPMScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import (
    is_torch_xla_available,
)
from diffusers.utils.torch_utils import randn_tensor
from pytorch_lightning import seed_everything

from measurements import DarkenOperator, get_noise

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# Load configurations
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def img_save(image, path):
    image = (image.detach() / 2 + 0.5).clamp(0, 1)
    image = (image.cpu().permute(0, 2, 3, 1)).squeeze(0).numpy()
    images = (image * 255).round().astype("uint8")
    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, images)


# import torch.nn.functional as F

class L2Loss(nn.Module):
    def forward(self, input, target):
        return torch.linalg.norm(input - target)
        # return torch.norm(torch.sum(torch.abs(input - target)), p=1, dim=0)
        # return torch.sqrt(nn.functional.mse_loss(input, target, reduction='mean')) * 255.
        # return torch.sqrt(nn.functional.mse_loss(input, target, reduction='sum'))
        # return torch.sqrt(torch.sum((input - target) ** 2))


class SmoothL1Loss(nn.Module):
    def forward(self, input, target):
        loss_function = nn.SmoothL1Loss(beta=1.0)
        return loss_function(input, target)


class L1Loss(nn.Module):
    def forward(self, input, target):
        loss_function = nn.L1Loss()
        return loss_function(input, target)


def blur_operation(input, scale=1):
    I1 = input
    # H, W = input.shape[-2:]
    for s in range(scale):
        # if s > 0:
        #     H, W = H//2, W//2
        #     I1 = kornia.geometry.transform.resize(I1, (H, W))
        I1 = kornia.filters.gaussian_blur2d(I1, (5, 5), (1.5, 1.5))
    return I1


class GuidedScheduler(
    # EulerAncestralDiscreteScheduler
    # DDIMScheduler
    DDPMScheduler
):
    # Copied from diffusers.schedulers.scheduling_repaint RePaintScheduler
    def reset_timesteps(
            self,
            num_inference_steps,
            timesteps_ref,
            device: Union[str, torch.device] = None,
            jump_length: int = 10,
            jump_n_sample: int = 10,
    ):
        if jump_length > 0:
            timesteps_ref = timesteps_ref.cpu().numpy()

            jumps = {}
            for j in range(num_inference_steps - 1, jump_length - 1, -jump_length):
                jumps[j] = jump_n_sample - 1
            timesteps = []
            i = -1
            while -1 <= i < (num_inference_steps - 2):
                i = i + 1
                timesteps.append(timesteps_ref[i])

                if jumps.get(i, 0) > 0:
                    jumps[i] = jumps[i] - 1
                    for _ in range(jump_length):
                        i = i - 1
                        if i > -1:
                            timesteps.append(timesteps_ref[i])

            self.timesteps_execution = torch.from_numpy(np.array(timesteps)).to(device)
        else:
            self.timesteps_execution = timesteps_ref

    def step_forward(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            sample: torch.FloatTensor,
    ):
        # TODO: for ddim scheduler
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        # # DDIM step_forward:
        sample = alpha_prod_t ** (0.5) * (
                (1 / alpha_prod_t_prev ** (0.5)) * sample +
                ((1 / alpha_prod_t - 1) ** (0.5) - (1 / alpha_prod_t_prev - 1) ** (0.5)) * model_output)

        # # DDIM step_backward:
        # sample = alpha_prod_t_prev ** (0.5) * \
        #               (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5) + \
        #               (1 - alpha_prod_t_prev) ** (0.5) * model_output

        return sample

    def undo_step(self, sample, timestep, generator=None):
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps

        for i in range(step_ratio):
            beta = self.betas[(timestep + i).cpu().long()]
            if sample.device.type == "mps":
                # randn does not work reproducibly on mps
                noise = randn_tensor(sample.shape, dtype=sample.dtype, generator=generator)
                noise = noise.to(sample.device)
            else:
                noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)

            # 10. Algorithm 1 Line 10 https://arxiv.org/pdf/2201.09865.pdf
            sample = (1 - beta) ** 0.5 * sample + beta ** 0.5 * noise

        # TODO: for euler scheduler
        if getattr(self, '_step_index', None) is not None:
            self._step_index -= 1

        return sample


class PLSDPipeline(StableDiffusionXLImg2ImgPipeline):
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            # image: PipelineImageInput = None,
            strength: float = 0.3,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            denoising_start: Optional[float] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Tuple[int, int] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Tuple[int, int] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            aesthetic_score: float = 6.0,
            negative_aesthetic_score: float = 2.5,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # callbacks
            # height: Optional[int] = 512,
            # width: Optional[int] = 512,
            do_measure: bool = True,
            jump_length: Optional[int] = 10,
            jump_n_sample: Optional[int] = 10,
            measurement: torch.Tensor = None,
            meas_inv: torch.Tensor = None,
            meas_inv_blur: torch.Tensor = None,
            operator=None,
            save_path: Optional[str] = None,
            step_size: Optional[float] = 10.0,
            w_inpaint: Optional[float] = 0.5,
            w_inv: Optional[float] = 0.5,
            w_inv_b: Optional[float] = 0.5,
            w_meas: Optional[float] = 0.5,
            w_stepsize_max: Optional[float] = 1.0,
            print_freq: Optional[int] = 10,
            use_adpative_stepsize=False,
            use_adpative_beta=False,
            **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:

        # TODO: Guidance
        set_requires_grad(self.text_encoder, False)
        set_requires_grad(self.vae, False)
        set_requires_grad(self.unet, False)
        self.loss_fn = L2Loss()  # SimplePerceptualLoss() #SmoothL1Loss() #L2Loss()
        self.loss_fn_latent = L2Loss()  # SmoothL1Loss() L2Loss() #nn.SmoothL1Loss(reduction='sum')

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            strength,
            num_inference_steps,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._denoising_start = denoising_start
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        # TODO: Guidance
        # 4. Preprocess image
        image = self.image_processor.preprocess(self.image_processor.denormalize(meas_inv_blur))

        # 5. Prepare timesteps
        def denoising_value_valid(dnv):
            return isinstance(dnv, float) and 0 < dnv < 1

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(
           num_inference_steps,
           strength,
           device,
           denoising_start=self.denoising_start if denoising_value_valid(self.denoising_start) else None,
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # # TODO: Time travel
        # self.scheduler.reset_timesteps(
        #     num_inference_steps=num_inference_steps, timesteps_ref=timesteps, device=device,
        #     jump_length=jump_length, jump_n_sample=jump_n_sample)
        #
        # print(f"timesteps (000): {self.scheduler.timesteps_execution[:500]}")
        # if len(self.scheduler.timesteps_execution) > 500:
        #     print(f"timesteps (-500): {self.scheduler.timesteps_execution[-500:]}")
        # latent_timestep = self.scheduler.timesteps_execution[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        init_latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            add_noise=False,
        )

        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.upcast_vae()

        name, ext = os.path.splitext(save_path)
        init_latents = init_latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        image_rec = self.vae.decode(
            init_latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image_rec = self.image_processor.postprocess(image_rec, output_type=output_type)[0]
        image_rec.save(f"{name}_image_init{ext}")

        noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        # get latents [original]
        latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)

        # # TODO: https://github.com/cloneofsimo/sdxl_inversions/blob/master/pnp_pipeline.py
        # # get latents [ddim]
        # latents = init_latents.detach()
        # added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        # if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        #     added_cond_kwargs["image_embeds"] = image_embeds
        # unet_kwargs['added_cond_kwargs'] = added_cond_kwargs
        # for t in self.progress_bar(reversed(self.scheduler.timesteps[-int(num_inference_steps*strength):])):
        #     noise_pred = self.unet(
        #         latents,
        #         t,
        #         encoder_hidden_states=prompt_embeds,
        #         cross_attention_kwargs=cross_attention_kwargs,
        #         added_cond_kwargs=added_cond_kwargs,
        #         return_dict=False,
        #     )[0]
        #     latents = self.scheduler.step_forward(noise_pred, t, latents)

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # height, width = image.shape[-2:]
        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 8. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 9.1 Apply denoising_end
        if (
                self.denoising_end is not None
                and self.denoising_start is not None
                and denoising_value_valid(self.denoising_end)
                and denoising_value_valid(self.denoising_start)
                and self.denoising_start >= self.denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {self.denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {self.denoising_end} when using type float."
            )
        elif self.denoising_end is not None and denoising_value_valid(self.denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)

        # TODO: Time travel
        # t_last = self.scheduler.timesteps_execution[0] + 1
        last_sample, last_grad = None, None

        # 10. Denoising loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # # TODO: Time travel
                # if t >= t_last:
                #     # compute the reverse: x_t-1 -> x_t
                #     latents = self.scheduler.undo_step(latents, t_last, generator)
                #     progress_bar.update()
                #     t_last = t
                #     continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                unet_kwargs = dict(
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    **unet_kwargs
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # TODO: Guidance
                if do_measure:
                    unet_kwargs['encoder_hidden_states'] = (
                        prompt_embeds.chunk(2)[1] if self.do_classifier_free_guidance else prompt_embeds
                    )
                    with torch.enable_grad():
                        latents_g = latents.detach().requires_grad_()
                        latent_model_input = self.scheduler.scale_model_input(latents_g, t)
                        # predict the noise residual
                        noise_pred_g = self.unet(latent_model_input, t, **unet_kwargs)[0]
                        sample = self.scheduler.step(
                            noise_pred_g, t, latents_g, **extra_step_kwargs, return_dict=True
                        ).pred_original_sample.to(device=noise_pred_g.device, dtype=noise_pred_g.dtype)
                        # TODO: for euler scheduler
                        if getattr(self.scheduler, '_step_index', None) is not None:
                            self.scheduler._step_index -= 1

                        image = self.vae.decode(
                            sample.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                            / self.vae.config.scaling_factor, return_dict=False)[0].to(dtype=latents_g.dtype)

                        loss = 0

                        meas_pred = operator.forward(image, is_latent=True, adaptive_beta=use_adpative_beta)
                        meas_pred = noiser(meas_pred)
                        origi_pred_blur = blur_operation(image, scale=2)
                        # measurement_blur = blur_operation(measurement, scale=2)
                        # meas_inv_blur = operator.transpose(measurement_blur, is_latent=True, adaptive_beta=use_adpative_beta)

                        meas_error = self.loss_fn(measurement, meas_pred)
                        if w_meas > 0:  # alpha
                            loss = loss + meas_error * w_meas

                        inv_error = self.loss_fn(meas_inv, image)
                        if w_inv > 0:  # omega
                            # w_inv=1.0
                            loss = loss + inv_error * w_inv

                        inv_error_b = self.loss_fn(meas_inv_blur, origi_pred_blur)
                        if w_inv_b > 0:  # omega
                            # w_inv=1.0
                            loss = loss + inv_error_b * w_inv_b

                        # ortho_pred = operator.transpose(operator.forward(origi_pred, is_latent=True), is_latent=True)
                        # ortho_project = origi_pred - ortho_pred
                        # parallel_project = operator.transpose(measurement, is_latent=True)
                        # inpainted_image = parallel_project + ortho_project
                        # encoded_z_0 = self.vae.config.scaling_factor * vae.encode(
                        #     # inpainted_image).latent_dist.mean
                        #     # meas_inv).latent_dist.sample(generator)
                        #     # meas_inv).latent_dist.mean
                        #     meas_inv_blur).latent_dist.mean
                        inpaint_error = self.loss_fn_latent(
                            init_latents,
                            self.vae.config.scaling_factor *
                            self.vae.encode(
                                origi_pred_blur.to(self.vae.encoder.conv_in.bias.dtype)
                            ).latent_dist.mean.to(dtype=latents_g.dtype)
                            # pred_original_sample, self.vae.config.scaling_factor *
                            # vae.encode(
                            # meas_inv.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                            # ).latent_dist.mean.to(dtype=latents_g.dtype)
                        )
                        if w_inpaint > 0:  # gamma
                            loss = loss + inpaint_error * w_inpaint

                        loss.backward()

                        # upgrade step_size
                        step_size_adp = step_size
                        if use_adpative_stepsize:
                            if not (t.item() >= 999 or last_sample is None or last_grad is None):
                                step_size_adp = torch.mul(sample - last_sample,
                                                          sample - last_sample).sum() / torch.mul(
                                    sample - last_sample, (latents_g.grad - last_grad)).sum()
                                step_size = max(min(step_size_adp, w_stepsize_max), 1 - 6)

                        if i % print_freq == 0:
                            print(f"[{i:04d}:{t}] lr_adp: {step_size_adp:.2f} loss: {loss.item():.2f} = "
                                  f"meas: {w_meas:.2f} * {0 if meas_error is None else meas_error.item():.2f} + "
                                  f"inv: {w_inv:.2f} * {0 if inv_error is None else inv_error.item():.2f} + "
                                  f"inv_b: {w_inv_b:.2f} * {0 if inv_error_b is None else inv_error_b.item():.2f} + "
                                  f"inpaint: {w_inpaint:.2f} * {0 if inpaint_error is None else inpaint_error.item():.2f}")
                            name, ext = os.path.splitext(save_path)
                            self.image_processor.postprocess(image.detach())[0].save(f"{name}_x0_pred_{t}{ext}")
                            self.image_processor.postprocess(origi_pred_blur.detach())[0].save(
                                f"{name}_origi_pred_blur_{t}{ext}")

                        assert self.scheduler.config.prediction_type == "epsilon"
                        with torch.no_grad():
                            # noise_pred_original -= self.scheduler.sigmas[self.scheduler.step_index] * (-latents.grad) * step_size
                            # noise_pred_original -= (1-self.scheduler.alphas_cumprod[timestep]) ** (0.5) * (-latents.grad) * step_size
                            latents -= latents_g.grad * step_size


                    last_sample = sample.detach()
                    last_grad = latents_g.grad.detach()
                    noise_pred = noise_pred.detach()
                    latents = latents.detach()

                    # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    add_neg_time_ids = callback_outputs.pop("add_neg_time_ids", add_neg_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

                # TODO: Time travel
                t_last = t

        if not output_type == "latent":
            if not needs_upcasting and latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            image = latents

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir to read samples from",
        default="lol_samples/"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        # default="8k uhd, soft lighting, high quality, best quality, extremely detailed, clean, tidy",
        help="the prompt to render"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="?",
        default="",
        # default="8k uhd, soft lighting, high quality, best quality, extremely detailed, clean, tidy",
        help="the negative_prompt to render"
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
        default=10,
        # default=999,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        # default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default='configs/LLIE_config_psld.yaml',
        help="task config yml file",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default='stabilityai/sdxl-turbo',
        # default='stabilityai/stable-diffusion-xl-base-1.0',
        # default='runwayml/stable-diffusion-v1-5',
        # default='stabilityai/stable-diffusion-2-1',
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
        "--lime_step",
        type=int,
        default=1,
        help="lime step",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=500,
        # L1Smooth:  0499] loss: 5000.00 = w_inpaint: 10000.00 * inpaint_error: 0.50
        # L2Sum: :0499] loss: 22.471407 = w_inpaint: 2.00 * inpaint_error: 11.235703
        # L1SmoothSum: 0499] loss: 3098.617676 = w_inpaint: 1.00 * inpaint_error: 3098.617676
        # L1SmoothSum: 0399] loss: 848.178406 = w_inpaint: 1.00 * inpaint_error: 848.178406
        help="inpainting error",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=200.0,
        # L1Smooth: 0899] loss: 2000.00 = w_inv: 200.00 * inv_error: 10.00
        # L2Sum: 0499] loss: 133.247955 = w_inv: 2.00 * inv_error: 66.623978
        help="measurement error",
    )
    parser.add_argument(
        "--omega_b",
        type=float,
        default=200.0,
        # L1Smooth: 0899] loss: 2000.00 = w_inv: 200.00 * inv_error: 10.00
        # L2Sum: 0499] loss: 133.247955 = w_inv: 2.00 * inv_error: 66.623978
        help="blurry measurement error",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        # L1Smooth: 0899] loss: 400.00 = w_meas: 4000.00 * meas_error: 0.10
        # L2Sum: 0459] loss: 33.642025 = w_meas: 2.00 * meas_error: 16.821012
        help="reconstruction error",
    )
    parser.add_argument(
        "--adp_stepsize",
        action='store_true',
        default=False,
        help="measurements beta adaptive",
    )
    parser.add_argument(
        "--adp_beta",
        action='store_true',
        default=False,
        help="measurements beta adaptive",
    )
    parser.add_argument(
        "--do_measure",
        action='store_true',
        default=False,
        help="measurements beta adaptive",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        # L1Smooth: 0899] loss: 400.00 = w_meas: 4000.00 * meas_error: 0.10
        # L2Sum: 0459] loss: 33.642025 = w_meas: 2.00 * meas_error: 16.821012
        help="adaptive step_size scale max value",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.3,
        help="adpative beta",
    )
    parser.add_argument(
        "--jump_len",
        type=int,
        default=0,
        help="time travel",
    )
    parser.add_argument(
        "--jump_sample",
        type=int,
        default=0,
        help="time travel",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="init strength",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--test_id",
        type=str,
        default="0",
        help="number of test id",
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weight_dtype = torch.float16

    pipeline = PLSDPipeline.from_pipe(
        StableDiffusionXLImg2ImgPipeline.from_pretrained(
            opt.model_id, dtype=weight_dtype, variant="fp16", use_safetensors=True
        )
    )
    # pipeline.scheduler = GuidedScheduler.from_config(
    #     pipeline.scheduler.config,
    #     timestep_spacing="trailing",
    # )
    # print(pipeline.scheduler.timesteps)

    pipeline.vae.to(device=torch_device, dtype=weight_dtype)
    # pipeline.vae.enable_xformers_memory_efficient_attention()
    pipeline.text_encoder.to(device=torch_device, dtype=weight_dtype)
    pipeline.text_encoder_2.to(device=torch_device, dtype=weight_dtype)
    pipeline.unet.to(device=torch_device, dtype=weight_dtype)
    # pipeline.unet.enable_xformers_memory_efficient_attention()

    pipeline.to(torch_device=torch_device)

    print(f"Done model loading")

    # exit(0)

    batch_size = opt.n_samples
    assert opt.prompt is not None

    outdir = os.path.join(opt.outdir, str(opt.test_id))
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'measurements'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'inter'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'results'), exist_ok=True)

    # df = pd.DataFrame(data=None, columns=['test_id', 'beta', 'PSNR', 'SSIM', 'LPIPS'])
    # df.to_csv("/home/eileen/Diffusion/DuduZhu/Diff_llie/psnr_ssim_lpips.csv", mode='a', index=False)

    imgsName = sorted(os.listdir(opt.indir))

    for i, imgName in enumerate(imgsName):

        #########################################################
        # set up operator and measurement
        #########################################################
        img_path = os.path.join(opt.indir, imgName)
        save_path = os.path.join(outdir, 'inter', f'inter_{imgName}')
        print('>>>>>>>>> ', str(opt.test_id), ' ', img_path, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # Prepare Operator and noise
        operator = DarkenOperator(
            device=torch_device, w=opt.W, h=opt.H,
            img_path=img_path,
            dtype=weight_dtype,
            # dtype=torch.float32,
            beta=opt.beta, sigma=opt.sigma,  # a=opt.a, b=opt.b
        )
        noiser = get_noise(name='gaussian', sigma=0.01)

        img = operator.origin_img
        if img.ndim == 2:
            img = img[:, :, None]

        measurement = torch.from_numpy(img.transpose((2, 0, 1))).contiguous().unsqueeze(0).to(device=torch_device,
                                                                                              dtype=weight_dtype)

        degradation_map = operator.get_illum(opt.lime_step).to(device=torch_device, dtype=weight_dtype)
        meas_inv = operator.transpose(measurement, is_latent=False, adaptive_beta=opt.adp_beta)
        meas_inv_blur = blur_operation(meas_inv, scale=2)

        vutils.save_image(measurement, os.path.join(outdir, 'measurements', f'L_{imgName}'))
        vutils.save_image(meas_inv, os.path.join(outdir, 'measurements', f'L_inv_{imgName}'))
        vutils.save_image(meas_inv_blur, os.path.join(outdir, 'measurements', f'L_inv_blur_{imgName}'))
        vutils.save_image(degradation_map, os.path.join(outdir, 'measurements', f'E0_{imgName}'))
        # vutils.save_image(L_img * E0_img ** -0.6, os.path.join(outdir, 'measurements', f'I0_{imgName}'))
        degradation_map = operator.get_illum(opt.lime_step).to(
            device=torch_device,
            # dtype=torch.float32,
            dtype=weight_dtype,
        )
        vutils.save_image(measurement * degradation_map ** (1 - torch.exp(degradation_map) - 0.7),
                          os.path.join(outdir, 'measurements', f'I0_{imgName}'))

        # operator.calculate_beta(L_img, 0.5)
        # operator.calculate_alpha(L_img)
        print(f"Done operator definition")

        #########################################################
        out = pipeline(
            prompt=opt.prompt,
            negative_prompt=opt.negative_prompt,
            # height=opt.H,
            # width=opt.W,
            num_inference_steps=opt.ddim_steps,
            guidance_scale=opt.scale,
            # output_type="pil",
            # return_dict=True,
            do_measure=opt.do_measure,
            jump_length=opt.jump_len,
            jump_n_sample=opt.jump_sample,
            strength=opt.strength,
            # callbacks
            measurement=measurement * 2.0 - 1.0,
            meas_inv=meas_inv * 2.0 - 1.0,
            meas_inv_blur=meas_inv_blur * 2.0 - 1.0,
            operator=operator,
            save_path=save_path,
            step_size=opt.step_size,
            w_inpaint=opt.gamma,
            w_inv=opt.omega,
            w_inv_b=opt.omega_b,
            w_meas=opt.alpha,
            w_stepsize_max=opt.eta,
            print_freq=opt.print_freq,
            use_adpative_stepsize=opt.adp_stepsize,
            use_adpative_beta=opt.adp_beta,
        ).images[0]

        out.save(os.path.join(outdir, 'results', f'{imgName}'))
        print('Done generation')

    # psnr, ssim, lpips = calculate_metrics(outdir)
    # output_list = [opt.test_id, opt.gamma, opt.omega, opt.alpha, psnr, ssim, lpips]
    # new_data = pd.DataFrame([output_list])
    # new_data.to_csv("psnr_ssim_lpips.csv", mode='a', header=False, index=False)

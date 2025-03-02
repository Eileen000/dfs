# Memo:
#  --prompt="8k uhd, soft lighting, high quality, best quality, extremely detailed, clean, tidy" \
#  --negative_prompt="art, blur, blurry, deformed iris, deformed pupils, dehydrated, disgusting, drawing, haze, low quality, semi-realistic, sketch, text, ugly, worst quality" \
#  --indir=data/size512 --outdir=outputs/psld-samples-llie --print_freq=100 \
#  --do_measure --adp_stepsize --step_size=0.5 --eta=1.0 --alpha=10.0 --omega=1.0 --omega_b=1.0 --gamma=1.0 \
#  --lime_step=1 --adp_beta --beta=0.8 --sigma=0.3 \
#  --model_id=stabilityai/sdxl-turbo --H=512 --W=512 --ddim_steps=50 --jump_len=0 --jump_sample=0 --strength=0.4 --scale=0.0 \
#  --model_id=stabilityai/stable-diffusion-xl-base-1.0 --H=512 --W=512 --ddim_steps=50 --jump_len=0 --jump_sample=0 --strength=0.5 --scale=10.5 \
#  --model_id=runwayml/stable-diffusion-v1-5 --H=512 --W=512 --ddim_steps=250  --jump_len=10 --jump_sample=10 --strength=0.5 --scale=7.5 \
#  --model_id=stabilityai/stable-diffusion-2-1 --H=512 --W=512 --ddim_steps=250  --jump_len=10 --jump_sample=10 --strength=0.5 --scale=7.5 \
#  --test_id=sdxltb-latent-p-lr0.5-a10-o1-o1b-g1g

# python new_sample.py \
#  --prompt="8k uhd, soft lighting, high quality, best quality, extremely detailed, clean, tidy" \
#  --negative_prompt="art, blur, blurry, deformed iris, deformed pupils, dehydrated, disgusting, drawing, haze, low quality, semi-realistic, sketch, text, ugly, worst quality" \ \
#  --indir=data/inputs --outdir=outputs/psld-samples-llie --print_freq=100 \
#  --do_measure --adp_stepsize --step_size=0.5 --eta=1.0 --alpha=10.0 --omega=1.0 --omega_b=1.0 --gamma=1.0 \
#  --lime_step=1 --adp_beta --beta=0.8 --sigma=0.3 \
#  --model_id=runwayml/stable-diffusion-v1-5 --H=256 --W=256 --ddim_steps=250  --jump_len=10 --jump_sample=5 --strength=0.5 \
#  --test_id=latent-p-lr0.5-a10-o1-o1b-g1g-tt10_5

# python new_sample_xl.py \
#   --prompt="8k uhd, soft lighting, high quality, best quality, extremely detailed, clean, tidy" \
#   --negative_prompt="art, blur, blurry, deformed iris, deformed pupils, dehydrated, disgusting, drawing, haze, low quality, semi-realistic, sketch, text, ugly, worst quality" \
#   --indir=data/size512 --outdir=outputs/psld-samples-llie --print_freq=5 \
#   --do_measure --adp_stepsize --step_size=1.0 --eta=1.0 --alpha=10.0 --omega=1.0 --omega_b=1.0 --gamma=1.0 \
#   --lime_step=1 --adp_beta --beta=0.8 --sigma=0.3 \
#   --model_id=stabilityai/sdxl-turbo --H=512 --W=512 --ddim_steps=50 --jump_len=0 --jump_sample=0 --strength=0.4 --scale=0.0 \
#   --test_id=sdxltb-latent-p-lr_adp-a10-o1-o1b-g1g

# python new_sample.py \
#   --prompt="8k uhd, soft lighting, high quality, best quality, extremely detailed, clean, tidy" \
#   --negative_prompt="art, blur, blurry, deformed iris, deformed pupils, dehydrated, disgusting, drawing, haze, low quality, semi-realistic, sketch, text, ugly, worst quality" \
#   --indir=lol_samples --outdir=outputs/psld-samples-llie --print_freq=5 \
#   --do_measure --adp_stepsize --eta=1.5 --alpha=1.0 --omega=1.0 --omega_b=0.5 --gamma=1.0 \
#   --lime_step=1 --adp_beta --sigma=0.3 \
#   --model_id=../stable-diffusion-v1-5 --H=512 --W=512 --ddim_steps=1000 --jump_len=0 --jump_sample=0 --strength=0.5 --scale=0.0 \
#   --test_id=test512



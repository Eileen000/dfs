# python paper_version_test2.py \
#   --indir=lolv1_512_512 \
#   --ddim_steps=999 --H=512 --W=512 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0 --step_size=0.0 --beta=0.0 \
#   --gb=0.0 --illum_constant=0.0 --rgb_equal=0.0 --color_direction=0.0 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=latent_rgb  #paper vision

# python paper_version_test2.py \
#   --indir=lolv1_512_512 \
#   --ddim_steps=999 --H=512 --W=512 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=10  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0 --step_size=2.0 --beta=0.0 \
#   --gb=0.0 --illum_constant=0.0 --rgb_equal=0.0 --color_direction=0.0 \
#   --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=gs2  #paper vision

# python paper_version_test2.py \
#   --indir=lolv1_512_512 \
#   --ddim_steps=999 --H=512 --W=512 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=10  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0 --step_size=5.0 --beta=0.0 \
#   --gb=0.0 --illum_constant=0.0 --rgb_equal=0.0 --color_direction=0.0 \
#   --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=gs5  #paper vision

# python paper_version_test2.py \
#   --indir=lolv1_512_512 \
#   --ddim_steps=999 --H=512 --W=512 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=10  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0 --step_size=50.0 --beta=0.0 \
#   --gb=0.0 --illum_constant=0.0 --rgb_equal=0.0 --color_direction=0.0 \
#   --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=gs50  #paper vision

python paper_version_test2.py \
  --indir=cat2 \
  --ddim_steps=999 --H=512 --W=512 --step=1 \
  --outdir=outputs/psld-samples-llie --prompt="" \
  --print_freq=100  \
  --gamma=1.0 --omega=1.0 --alpha=1.0 --step_size=0.0 --beta=0.0 \
  --gb=0.0 --illum_constant=0.0 --rgb_equal=0.0 --color_direction=0.0 \
  --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
  --jump_len=0 --jump_sample=0 \
  --a=0.0 \
  --test_id=cat  #paper vision
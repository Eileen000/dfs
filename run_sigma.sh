# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.01  --step_size=1.0 --beta=0.79 \
#   --test_id=601

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=0.5 --omega=1.0 --alpha=0.01  --step_size=1.0 --beta=0.79 \
#   --test_id=602

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.01  --step_size=1.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5 \
#   --test_id=603
python psld_low_light_adaptive.py \
  --ddim_steps=999 --H=256 --W=256 --step=1 \
  --outdir=outputs/psld-samples-llie --prompt="" \
  --print_freq=100  \
  --gamma=1.0 --omega=1.0 --alpha=0.02  --step_size=0.0 --beta=0.79 \
  --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
  --test_id=1117

python psld_low_light_adaptive.py \
  --ddim_steps=999 --H=256 --W=256 --step=1 \
  --outdir=outputs/psld-samples-llie --prompt="" \
  --print_freq=100  \
  --gamma=1.0 --omega=1.0 --alpha=0.05  --step_size=0.0 --beta=0.79 \
  --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
  --test_id=1118

python psld_low_light_adaptive.py \
  --ddim_steps=999 --H=256 --W=256 --step=1 \
  --outdir=outputs/psld-samples-llie --prompt="" \
  --print_freq=100  \
  --gamma=1.0 --omega=1.0 --alpha=0.1  --step_size=0.0 --beta=0.79 \
  --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
  --test_id=1119

  # --adp_beta--sigma=0.30 \
  # --adp_stepsize --eta=0.5 \
  

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.01  --step_size=1.0 --beta=0.75 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.30 \
#   --jump_len=10 --jump_sample=10 \
#   --test_id=403

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.01  --step_size=1.0 --beta=0.75 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.24 \
#   --test_id=111

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.01  --step_size=1.0 --beta=0.75 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.28 \
#   --test_id=112


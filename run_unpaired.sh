# python paper_version.py \
#   --indir=dicm_samples \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=1 #dicm

# python paper_version.py \
#   --indir=lime_samples \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=2  #lime

# python paper_version.py \
#   --indir=mef_samples \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=3  #mef

# python paper_version.py \
#   --indir=npe_samples \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=4  #npe

# python paper_version.py \
#   --indir=vv_samples \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=5  #vv

python paper_version.py \
  --indir=lol_v2_samples \
  --ddim_steps=999 --H=256 --W=256 --step=1 \
  --outdir=outputs/psld-samples-llie --prompt="" \
  --print_freq=100  \
  --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
  --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
  --jump_len=0 --jump_sample=0 \
  --a=0.0 \
  --test_id=6  #lol v2 synthsis
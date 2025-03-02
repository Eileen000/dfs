python paper_version_test.py \
  --indir=LOL_V2_test_params \
  --ddim_steps=600 --H=512 --W=512 --step=1 \
  --outdir=outputs/psld-samples-llie --prompt="" \
  --print_freq=100  \
  --gamma=10.0 --omega=20.0 --alpha=15.0  --step_size=0.0 --beta=0.0 \
  --adp_stepsize --eta=0.5  --adp_beta --sigma=0.275 \
  --jump_len=0 --jump_sample=0 \
  --a=0.0 \
  --test_id=test_t600_g10.0_o20.0_a15.0_e0.5_s0.275  #paper vision

python paper_version_test.py \
  --indir=LOL_V2_test_params \
  --ddim_steps=600 --H=512 --W=512 --step=1 \
  --outdir=outputs/psld-samples-llie --prompt="" \
  --print_freq=100  \
  --gamma=10.0 --omega=20.0 --alpha=15.0  --step_size=0.0 --beta=0.0 \
  --adp_stepsize --eta=0.7  --adp_beta --sigma=0.275 \
  --jump_len=0 --jump_sample=0 \
  --a=0.0 \
  --test_id=test_t600_g10.0_o20.0_a15.0_e0.7_s0.275  #paper vision

python paper_version_test.py \
  --indir=LOL_V2_test_params \
  --ddim_steps=600 --H=512 --W=512 --step=1 \
  --outdir=outputs/psld-samples-llie --prompt="" \
  --print_freq=100  \
  --gamma=10.0 --omega=20.0 --alpha=15.0  --step_size=0.0 --beta=0.0 \
  --adp_stepsize --eta=0.6  --adp_beta --sigma=0.275 \
  --jump_len=0 --jump_sample=0 \
  --a=0.0 \
  --test_id=test_t600_g10.0_o20.0_a15.0_e0.6_s0.275  #paper vision
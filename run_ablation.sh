# python psld_low_light_adaptive_.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.01  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
#   --test_id=618

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.2  --step_size=0.0 --beta=0.79 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
#   --test_id=1120

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.5  --step_size=0.0 --beta=0.79 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
#   --test_id=1121

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.8  --step_size=0.0 --beta=0.79 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
#   --test_id=1122


# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.79 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
#   --test_id=1123

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.2  --step_size=0.005 --beta=0.79 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
#   --test_id=1120

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.2  --step_size=0.001 --beta=0.79 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
#   --test_id=1120

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5 --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --test_id=1137

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=10.0 --b=1.0 \
#   --test_id=2209

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=10.0 --b=1.0 \
#   --test_id=2210  #inv term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=10.0 --b=1.0 \
#   --test_id=2214  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=1.0 \
#   --test_id=2215  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=2.0 \
#   --test_id=2216  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=3.0 \
#   --test_id=2217  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=4.0 \
#   --test_id=2218  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=5.0 \
#   --test_id=2219  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=6.0 \
#   --test_id=2220  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=7.0 \
#   --test_id=2221  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=8.0 \
#   --test_id=2222  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=9.0 \
#   --test_id=2223  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=10.0 \
#   --test_id=2224  #inpaint term : only invdenoised

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=11.0 \
#   --test_id=2225  #inpaint term : only invdenoised


# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=0.0 --omega=3.0 --alpha=0.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=3.0 \
#   --test_id=2301  #inpaint term : only invdenoised only omega


# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=0.0 --omega=3.0 --alpha=3.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=3.0 \
#   --test_id=2303  #inpaint term : only invdenoised only omega


# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=3.0 --omega=3.0 --alpha=0.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=3.0 \
#   --test_id=2305  #inpaint term : only invdenoised only omega

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=3.0 \
#   --test_id=2306  # invG1

# python psld_low_light_adaptive.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=3.0 \
#   --test_id=2307  # invG1

# python paper_version.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=1  

####  gamma: latent    omega: inverse    alpha: meas
# python paper_version.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=0.0 --omega=0.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=1  #

# python paper_version.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=0.0 --omega=1.0 --alpha=0.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=2  #

# python paper_version.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=0.0 --alpha=0.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=3  #

# python paper_version.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=0.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=4  #

# python paper_version.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=0.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=5  #

# python paper_version.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=0.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=6  #

# python paper_version.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
#   --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=7  #

# python paper_version.py \
#   --ddim_steps=999 --H=256 --W=256 --step=1 \
#   --outdir=outputs/psld-samples-llie --prompt="" \
#   --print_freq=100  \
#   --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.7 \
#   --adp_stepsize --eta=0.5 \
#   --jump_len=0 --jump_sample=0 \
#   --a=0.0 \
#   --test_id=12  #

python paper_version.py \
  --indir=dicm_samples \
  --ddim_steps=999 --H=256 --W=256 --step=1 \
  --outdir=outputs/psld-samples-llie --prompt="" \
  --print_freq=100  \
  --gamma=1.0 --omega=1.0 --alpha=1.0  --step_size=0.0 --beta=0.0 \
  --adp_stepsize --eta=0.5  --adp_beta --sigma=0.3 \
  --jump_len=0 --jump_sample=0 \
  --a=0.0 \
  --test_id=1  #paper vision
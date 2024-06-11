'''
python scripts/sampling/pnp_generate_config.py \
    --p_config config_pnp_auto.yaml \
    --output_path "outputs/automatic_ref_editing/image" \
    --image_path "outputs/centerframe/tshirtman.png" \
    --latents_path "outputs/automatic_ref_editing/latents_forward" \
    --prompt "a man walks on the beach" 
'''


import yaml
import argparse

def save_yaml(args):
    config_data = {
            'seed': args.seed,
            'device': args.device,
            'output_path': args.output_path,
            'image_path': args.image_path,
            'latents_path': args.latents_path,
            'sd_version': args.sd_version,
            'guidance_scale': args.guidance_scale,
            'n_timesteps': args.n_timesteps,
            'prompt': args.prompt,
            'negative_prompt': args.negative_prompt,
            'pnp_attn_t': args.pnp_attn_t,
            'pnp_f_t': args.pnp_f_t
    }
    
    with open(args.p_config, 'w') as file:
        yaml.dump(config_data, file, sort_keys=False, allow_unicode=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save configuration to a YAML file.")
    parser.add_argument('--p_config', type=str, help="Path to save the YAML configuration file.")
    parser.add_argument('--output_path', type=str, help="Output path for the results.")
    parser.add_argument('--image_path', type=str, help="Path to the input image.")
    parser.add_argument('--latents_path', type=str, help="Path to the latents file.")
    parser.add_argument('--prompt', type=str, help="Prompt for the diffusion model.")
    parser.add_argument('--seed', type=int, default=1, help="Seed for random number generation.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to be used (e.g., 'cuda', 'cpu').")
    parser.add_argument('--sd_version', type=str, default='2.1', help="Version of the diffusion model.")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help="Guidance scale for the diffusion model.")
    parser.add_argument('--n_timesteps', type=int, default=50, help="Number of timesteps for the diffusion process.")
    parser.add_argument('--negative_prompt', type=str, default='ugly, blurry, black, low res, unrealistic', help="Negative prompt for the diffusion model.")
    parser.add_argument('--pnp_attn_t', type=float, default=0.5, help="PNP attention threshold.")
    parser.add_argument('--pnp_f_t', type=float, default=0.8, help="PNP feature threshold.")
    
    args = parser.parse_args()
    
    save_yaml(args)
    print(f"YAML configuration saved to {args.p_config}")

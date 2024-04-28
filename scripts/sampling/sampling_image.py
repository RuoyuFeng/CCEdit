from pytorch_lightning import seed_everything
from scripts.demo.streamlit_helpers import *
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering

import argparse
import tqdm

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', type=str, default='2.1', 
                        choices=['2.1', '2.1-768', 'xl'])
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="a corgi is sitting on a couch")
    parser.add_argument("--prompt_listpath", type=str, default="", help="path to a txt file with a list of prompts")
    parser.add_argument("--negative_prompt", type=str, default="ugly, low quality")
    parser.add_argument('--save_path', type=str, default='outputs/demo/txt2img/')
    args = parser.parse_args()

    seed_everything(args.seed)
    save_path = args.save_path

    version_map = {
        '2.1': 'sd-2.1',
        '2.1-768': 'sd-2.1-768',
        'xl': 'SD-XL base',
    }

    SD_XL_BASE_RATIOS = {
        "0.5": (704, 1408),
        "0.52": (704, 1344),
        "0.57": (768, 1344),
        "0.6": (768, 1280),
        "0.68": (832, 1216),
        "0.72": (832, 1152),
        "0.78": (896, 1152),
        "0.82": (896, 1088),
        "0.88": (960, 1088),
        "0.94": (960, 1024),
        "1.0": (1024, 1024),
        "1.07": (1024, 960),
        "1.13": (1088, 960),
        "1.21": (1088, 896),
        "1.29": (1152, 896),
        "1.38": (1152, 832),
        "1.46": (1216, 832),
        "1.67": (1280, 768),
        "1.75": (1344, 768),
        "1.91": (1344, 704),
        "2.0": (1408, 704),
        "2.09": (1472, 704),
        "2.4": (1536, 640),
        "2.5": (1600, 640),
        "2.89": (1664, 576),
        "3.0": (1728, 576),
    }

    VERSION2SPECS = {
        "SD-XL base": {
            "H": 1024,
            "W": 1024,
            "C": 4,
            "f": 8,
            "is_legacy": False,
            "config": "configs/inference/sd_xl_base.yaml",
            "ckpt": "checkpoints/sd_xl_base_0.9.safetensors",
            "is_guided": True,
        },
        "sd-2.1": {
            "H": 512,
            "W": 512,
            "C": 4,
            "f": 8,
            "is_legacy": True,
            "config": "configs/inference/sd_2_1.yaml",
            "ckpt": "checkpoints/v2-1_512-ema-pruned.safetensors",
            "is_guided": True,
        },
        "sd-2.1-768": {
            "H": 768,
            "W": 768,
            "C": 4,
            "f": 8,
            "is_legacy": True,
            "config": "configs/inference/sd_2_1_768.yaml",
            "ckpt": "checkpoints/v2-1_768-ema-pruned.safetensors",
        },
        "SDXL-Refiner": {
            "H": 1024,
            "W": 1024,
            "C": 4,
            "f": 8,
            "is_legacy": True,
            "config": "configs/inference/sd_xl_refiner.yaml",
            "ckpt": "checkpoints/sd_xl_refiner_0.9.safetensors",
            "is_guided": True,
        },
    }

    version = args.model_version
    version = version_map[version]
    version_dict = VERSION2SPECS[version]

    # initialize model
    state = init_st(version_dict)
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]

    if version == "SD-XL base":
        ratio = '1.0'
        W, H = SD_XL_BASE_RATIOS[ratio]
    else:
        W, H = version_dict['W'], version_dict['H']

    C = version_dict["C"]
    F = version_dict["f"]

    if args.prompt_listpath:
        with open(args.prompt_listpath, 'r') as f:
            prompts = f.readlines()
        prompts = [p.strip() for p in prompts]
    else:
        prompts = [args.prompt]
    negative_prompt = args.negative_prompt
    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }

    for prompt in tqdm.tqdm(prompts):
        print('Current Prompt: >>>>> {} <<<<<'.format(prompt))
        value_dict = init_embedder_options(
            get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
            init_dict,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        _, _, sampler = init_sampling(
            use_identity_guider=not version_dict["is_guided"]
        )

        num_samples = args.num_samples

        is_legacy=False
        return_latents = False
        filter=None
        with torch.no_grad():
            samples = do_sample(
                state["model"],
                sampler,
                value_dict,
                num_samples,
                H,
                W,
                C,
                F,
                force_uc_zero_embeddings=["txt"] if not is_legacy else [],
                return_latents=return_latents,
                filter=filter,
            )

        if samples is not None:
            perform_save_locally(save_path, samples)
            print("Saved samples to {}. Enjoy.".format(save_path))

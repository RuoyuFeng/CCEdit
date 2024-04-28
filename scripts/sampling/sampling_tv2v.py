import argparse
import json
import os
import random

import torch
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from safetensors import safe_open
from torch import autocast

from scripts.sampling.util import (
    chunk,
    convert_load_lora,
    create_model,
    init_sampling,
    load_video_keyframes,
    model_load_ckpt,
    perform_save_locally_video,
)
from sgm.util import append_dims

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--use_default", action="store_true", help="use default ckpt at first"
    )
    parser.add_argument(
        "--basemodel_path",
        type=str,
        default="",
        help="load a new base model instead of original sd-1.5",
    )
    parser.add_argument("--basemodel_listpath", type=str, default="")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--vae_path", type=str, default="")
    parser.add_argument(
        "--video_path",
        type=str,
        default="",
    )
    parser.add_argument("--prompt_listpath", type=str, default="")
    parser.add_argument("--video_listpath", type=str, default="")
    parser.add_argument(
        "--videos_directory",
        type=str,
        default="",
        help="directory containing videos to be processed",
    )
    parser.add_argument(
        '--json_path',
        type=str,
        default='',
        help='path to json file containing video paths and captions'
    )
    parser.add_argument(
        '--videos_root',
        type=str,
        default='',
        help='path to the root of videos'
    )
    parser.add_argument("--save_path", type=str, default="outputs/demo/tv2v")
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=384)
    parser.add_argument("--detect_ratio", type=float, default=1.0)
    parser.add_argument("--original_fps", type=int, default=20)
    parser.add_argument("--target_fps", type=int, default=3)
    parser.add_argument("--num_keyframes", type=int, default=9)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="ugly, low quality")
    parser.add_argument("--add_prompt", type=str, default="masterpiece, high quality")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sampler_name", type=str, default="EulerEDMSampler")
    parser.add_argument(
        "--discretization_name", type=str, default="LegacyDDPMDiscretization"
    )
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--prior_coefficient_x", type=float, default=0.0)
    parser.add_argument("--prior_coefficient_noise", type=float, default=1.0)
    parser.add_argument("--sdedit_denoise_strength", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--disable_check_repeat', action='store_true', help='disable check repeat')
    parser.add_argument('--lora_strength', type=float, default=0.8)
    parser.add_argument('--save_type', type=str, default='mp4', choices=['gif', 'mp4'])
    parser.add_argument('--inpainting_mode', action='store_true', help='inpainting mode')
    args = parser.parse_args()

    seed = args.seed
    if seed == -1:
        seed = random.randint(0, 1000000)
    seed_everything(seed)

    # initialize the model
    model = create_model(config_path=args.config_path).to("cuda")
    ckpt_path = args.ckpt_path
    print("--> load ckpt from: ", ckpt_path)
    model = model_load_ckpt(model, path=ckpt_path)
    model.eval()

    # load the prompts and video_paths
    video_save_paths = []
    assert not (args.prompt_listpath and args.videos_directory), (
        "Only one of prompt_listpath and videos_directory can be provided, "
        "but got prompt_listpath: {}, videos_directory: {}".format(
            args.prompt_listpath, args.videos_directory
        )
    )
    if args.prompt_listpath:
        with open(args.prompt_listpath, "r") as f:
            prompts = f.readlines()
        prompts = [p.strip() for p in prompts]
        # load paths of cond_img
        assert args.video_listpath, (
            "video_listpath must be provided when prompt_listpath is provided, "
            "but got video_listpath: {}".format(args.video_listpath)
        )
        with open(args.video_listpath, "r") as f:
            video_paths = f.readlines()
        video_paths = [p.strip() for p in video_paths]
    elif args.videos_directory:
        prompts = []
        video_paths = []
        for video_name in os.listdir(args.videos_directory):
            video_path = os.path.join(args.videos_directory, video_name)
            if os.path.isdir(video_path):
                prompts.append(video_name)
                video_paths.append(video_path)
    elif args.json_path:
        assert args.videos_root != '', 'videos_root must be provided when json_path is provided'
        with open(args.json_path, 'r') as f:
            json_dict = json.load(f)
        prompts = []
        video_paths = []
        for item in json_dict:
            video_path = os.path.join(args.videos_root, item["Video Type"], item["Video Name"] + '.mp4')
            
            for edit in item['Editing']:
                video_paths.append(video_path)
                prompts.append(edit["Target Prompt"])
                video_save_paths.append(
                    os.path.join(args.save_path, item["Video Type"], item["Video Name"], edit["Target Prompt"])
                )
    else:
        assert args.prompt and args.video_path, (
            "prompt and video_path must be provided when prompt_listpath and videos_directory are not provided, "
            "but got prompt: {}, video_path: {}".format(args.prompt, args.video_path)
        )
        prompts = [args.prompt]
        video_paths = [args.video_path]

    assert len(prompts) == len(
        video_paths
    ), "The number of prompts and video_paths must be the same, and you provided {} prompts and {} video_paths".format(
        len(prompts), len(video_paths)
    )
    num_samples = args.num_samples
    batch_size = args.batch_size

    print("\nNumber of prompts: {}".format(len(prompts)))
    print("Generate {} samples for each prompt".format(num_samples))

    prompts = [item for item in prompts for _ in range(num_samples)]
    video_paths = [item for item in video_paths for _ in range(num_samples)]

    prompts_chunk = list(chunk(prompts, batch_size))
    video_paths_chunk = list(chunk(video_paths, batch_size))
    del prompts
    del video_paths

    # load paths of basemodel if provided
    assert not (args.basemodel_path and args.basemodel_listpath), (
        "Only one of basemodel_path and basemodel_listpath can be provided, "
        "but got basemodel_path: {}, basemodel_listpath: {}".format(
            args.basemodel_path, args.basemodel_listpath
        )
    )
    basemodel_paths = []
    if args.basemodel_listpath:
        with open(args.basemodel_listpath, "r") as f:
            basemodel_paths = f.readlines()
        basemodel_paths = [p.strip() for p in basemodel_paths]
    if args.basemodel_path:
        basemodel_paths = [args.basemodel_path]
    if args.use_default:
        basemodel_paths = ["default"] + basemodel_paths
    if len(basemodel_paths) == 0:
        basemodel_paths = ["default"]

    for basemodel_idx, basemodel_path in enumerate(basemodel_paths):
        print("-> base model idx: ", basemodel_idx)
        print("-> base model path: ", basemodel_path)

        if basemodel_path == "default":
            pass
        elif basemodel_path:
            print("--> load a new base model from {}".format(basemodel_path))
            model = model_load_ckpt(model, basemodel_path, True)

        if args.lora_path:
            print("--> load a new LoRA model from {}".format(args.lora_path))
            sd_state_dict = model.state_dict()
            lora_path = args.lora_path

            if lora_path.endswith(".safetensors"):
                lora_state_dict = {}

                # with safe_open(lora_path, framework="pt", device='cpu') as f:
                with safe_open(lora_path, framework="pt", device=0) as f:
                    for key in f.keys():
                        lora_state_dict[key] = f.get_tensor(key)

                is_lora = all("lora" in k for k in lora_state_dict.keys())
                if not is_lora:
                    raise ValueError(
                        f"The model you provided in [{lora_path}] is not a LoRA model. "
                    )
            else:
                raise NotImplementedError
            sd_state_dict = convert_load_lora(
                sd_state_dict, lora_state_dict, alpha=args.lora_strength
            )  #
            model.load_state_dict(sd_state_dict)

        # TODO: the logic here is not elegant.
        if args.vae_path:
            vae_path = args.vae_path
            print("--> load a new VAE model from {}".format(vae_path))

            if vae_path.endswith(".pt"):
                vae_state_dict = torch.load(vae_path, map_location="cpu")["state_dict"]
                msg = model.first_stage_model.load_state_dict(
                    vae_state_dict, strict=False
                )
            elif vae_path.endswith(".safetensors"):
                vae_state_dict = {}

                # with safe_open(vae_path, framework="pt", device='cpu') as f:
                with safe_open(vae_path, framework="pt", device=0) as f:
                    for key in f.keys():
                        vae_state_dict[key] = f.get_tensor(key)

                msg = model.first_stage_model.load_state_dict(
                    vae_state_dict, strict=False
                )
            else:
                raise ValueError("Cannot load vae model from {}".format(vae_path))

            print("msg of loading vae: ", msg)

        if os.path.exists(
            os.path.join(
                args.save_path,
                basemodel_path.split("/")[-1].split(".")[0],
                "log_info.json",
            )
        ):
            with open(
                os.path.join(
                    args.save_path,
                    basemodel_path.split("/")[-1].split(".")[0],
                    "log_info.json",
                ),
                "r",
            ) as f:
                log_info = json.load(f)
        else:
            log_info = {
                "basemodel_path": basemodel_path,
                "lora_path": args.lora_path,
                "vae_path": args.vae_path,
                "video_paths": [],
                "keyframes_paths": [],
            }

        num_keyframes = args.num_keyframes

        for idx, (prompts, video_paths) in enumerate(
            zip(prompts_chunk, video_paths_chunk)
        ):
            # if idx == 2: # ! DEBUG
            #     break
            if not args.disable_check_repeat:
                while video_paths[0] in log_info["video_paths"]:
                    print(f"video [{video_paths[0]}] has been processed, skip it.")
                    prompts_list, video_paths_list = list(prompts), list(video_paths)
                    prompts_list.pop(0)
                    video_paths_list.pop(0)
                    prompts, video_paths = tuple(prompts_list), tuple(video_paths_list)
                    del prompts_list, video_paths_list
                    if len(prompts) == 0:
                        break
                if len(video_paths) == 0:
                    continue

            bs = min(len(prompts), batch_size)
            print(f"\nProgress: {idx} / {len(prompts_chunk)}. ")
            H, W = args.H, args.W
            keyframes_list = []
            print("load video ...")
            try:
                for video_path in video_paths:
                    keyframes = load_video_keyframes(
                        video_path,
                        args.original_fps,
                        args.target_fps,
                        num_keyframes,
                        (H, W),
                    )
                    keyframes = keyframes.unsqueeze(0)  # B T C H W
                    keyframes = rearrange(keyframes, "b t c h w -> b c t h w").to(
                        model.device
                    )
                    keyframes_list.append(keyframes)
            except:
                print(f"Error when loading video from  {video_paths}")
                continue
            print("load video done ...")
            keyframes = torch.cat(keyframes_list, dim=0)
            control_hint = keyframes

            batch = {
                "txt": prompts,
                "control_hint": control_hint,
            }

            negative_prompt = args.negative_prompt
            batch_uc = {
                "txt": [negative_prompt for _ in range(bs)],
                "control_hint": batch[
                    "control_hint"
                ].clone(),  # to use the pretrained weights, we must use the same control_hint in the batch_uc
            }
            # batch["txt"] = ["masterpiece, best quality, " + each for each in batch["txt"]]
            if args.add_prompt:
                batch["txt"] = [args.add_prompt + ", " + each for each in batch["txt"]]
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch_c=batch,
                batch_uc=batch_uc,
            )

            sampling_kwargs = {}  # usually empty

            for k in c:
                if isinstance(c[k], torch.Tensor):
                    c[k], uc[k] = map(lambda y: y[k][:bs].to(model.device), (c, uc))
            shape = (4, num_keyframes, H // 8, W // 8)

            precision_scope = autocast
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    randn = torch.randn(bs, *shape).to(model.device)
                    if args.sdedit_denoise_strength == 0.0:

                        def denoiser(input, sigma, c):
                            return model.denoiser(
                                model.model, input, sigma, c, **sampling_kwargs
                            )

                        if args.prior_coefficient_x != 0.0:
                            prior = model.encode_first_stage(keyframes)
                            randn = (
                                args.prior_coefficient_x * prior
                                + args.prior_coefficient_noise * randn
                            )
                        sampler = init_sampling(
                            sample_steps=args.sample_steps,
                            sampler_name=args.sampler_name,
                            discretization_name=args.discretization_name,
                            guider_config_target="sgm.modules.diffusionmodules.guiders.VanillaCFGTV2V",
                            cfg_scale=args.cfg_scale,
                        )
                        sampler.verbose = True
                        if args.inpainting_mode:
                            raise NotImplementedError
                            # TODO: this is a temporary debug code, mask should be provided by the user
                            # mask = torch.zeros_like(uc['control_hint'])
                            mask = torch.zeros_like(c['control_hint'])
                            mask[-c['control_hint'] == -1] = 1
                            mask = torch.nn.functional.interpolate(mask, size=(mask.shape[2], H // 8, W // 8), mode='area')
                            mask = mask[:,1].unsqueeze(1)
                            mask = torch.round(mask)
                            mask = torch.clamp(mask, 0, 1)
                            z = model.encode_first_stage(keyframes)
                            # import pdb; pdb.set_trace()
                            # import torchvision, einops
                            # torchvision.utils.save_image(einops.rearrange(-uc['control_hint'], 'b c t h w -> (b t) c h w'), 'debug_inpaint_hint.png', normalize=True, range=(-1, 1))
                            # torchvision.utils.save_image(einops.rearrange(mask, 'b c t h w -> (b t) c h w'), 'debug_mask.png', normalize=True, range=(-1, 1))
                            uc['control_hint'] = c['control_hint'].clone()  # TODO: debug, must delete
                            # import pdb; pdb.set_trace()
                            # import torchvision, einops
                            # torchvision.utils.save_image(einops.rearrange(-uc['control_hint'], 'b c t h w -> (b t) c h w'),'debug_uc.png', normalize=True, range=(-1, 1))
                            # torchvision.utils.save_image(einops.rearrange(-c['control_hint'], 'b c t h w -> (b t) c h w'),'debug_c.png', normalize=True, range=(-1, 1))
                            # torchvision.utils.save_image(einops.rearrange(mask, 'b c t h w -> (b t) c h w'),'debug_mask.png', normalize=True, range=(-1, 1))

                            samples = sampler.sample_inpainting(denoiser, randn, c, uc=uc, x0=z, mask=mask)
                        else:                        
                            samples = sampler(denoiser, randn, c, uc=uc)
                    else:
                        assert (
                            args.sdedit_denoise_strength > 0.0
                        ), "sdedit_denoise_strength should be positive"
                        assert (
                            args.sdedit_denoise_strength <= 1.0
                        ), "sdedit_denoise_strength should be less than 1.0"
                        assert (
                            args.prior_coefficient_x == 0
                        ), "prior_coefficient_x should be 0 when using sdedit_denoise_strength"
                        denoise_strength = args.sdedit_denoise_strength
                        sampler = init_sampling(
                            sample_steps=args.sample_steps,
                            sampler_name=args.sampler_name,
                            discretization_name=args.discretization_name,
                            guider_config_target="sgm.modules.diffusionmodules.guiders.VanillaCFGTV2V",
                            cfg_scale=args.cfg_scale,
                            img2img_strength=denoise_strength,
                        )
                        sampler.verbose = True
                        z = model.encode_first_stage(keyframes)
                        noise = torch.randn_like(z)
                        sigmas = sampler.discretization(sampler.num_steps).to(z.device)
                        sigma = sigmas[0]

                        print(f"all sigmas: {sigmas}")
                        print(f"noising sigma: {sigma}")
                        noised_z = z + noise * append_dims(sigma, z.ndim)
                        noised_z = noised_z / torch.sqrt(
                            1.0 + sigmas[0] ** 2.0
                        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

                        def denoiser(x, sigma, c):
                            return model.denoiser(model.model, x, sigma, c)
                        if args.inpainting_mode:
                            raise NotImplementedError
                            # TODO: this is a temporary debug code, mask should be provided by the user
                            # mask = torch.zeros_like(uc['control_hint'])
                            mask = torch.zeros_like(c['control_hint'])
                            mask[-c['control_hint'] == -1] = 1
                            mask = torch.nn.functional.interpolate(mask, size=(mask.shape[2], H // 8, W // 8), mode='area')
                            mask = mask[:,1].unsqueeze(1)
                            mask = torch.round(mask)
                            mask = torch.clamp(mask, 0, 1)
                            z = model.encode_first_stage(keyframes)
                            # import pdb; pdb.set_trace()
                            # import torchvision, einops
                            # torchvision.utils.save_image(einops.rearrange(-uc['control_hint'], 'b c t h w -> (b t) c h w'), 'debug_inpaint_hint.png', normalize=True, range=(-1, 1))
                            # torchvision.utils.save_image(einops.rearrange(mask, 'b c t h w -> (b t) c h w'), 'debug_mask.png', normalize=True, range=(-1, 1))
                            uc['control_hint'] = c['control_hint'].clone()  # TODO: debug, must delete
                            # import pdb; pdb.set_trace()
                            # import torchvision, einops
                            # torchvision.utils.save_image(einops.rearrange(-uc['control_hint'], 'b c t h w -> (b t) c h w'),'debug_uc.png', normalize=True, range=(-1, 1))
                            # torchvision.utils.save_image(einops.rearrange(-c['control_hint'], 'b c t h w -> (b t) c h w'),'debug_c.png', normalize=True, range=(-1, 1))
                            # torchvision.utils.save_image(einops.rearrange(mask, 'b c t h w -> (b t) c h w'),'debug_mask.png', normalize=True, range=(-1, 1))

                            samples = sampler.sample_inpainting(denoiser, noised_z, c, uc=uc, x0=z, mask=mask)
                        else:
                            samples = sampler(denoiser, noised_z, cond=c, uc=uc)

                    samples = model.decode_first_stage(samples)

            # save the results
            keyframes = (torch.clamp(keyframes, -1.0, 1.0) + 1.0) / 2.0
            samples = (torch.clamp(samples, -1.0, 1.0) + 1.0) / 2.0
            control_hint = (torch.clamp(c["control_hint"], -1.0, 1.0) + 1.0) / 2.0
            if video_save_paths == []:
                save_path = args.save_path
                save_path = os.path.join(
                    save_path, basemodel_path.split("/")[-1].split(".")[0]
                )
            else:
                save_path = video_save_paths[idx]

            perform_save_locally_video(
                os.path.join(save_path, "original"), 
                keyframes, 
                args.target_fps, 
                args.save_type,
                save_grid=False
            )

            keyframes_paths = perform_save_locally_video(
                os.path.join(save_path, "result"),
                samples,
                args.target_fps,
                args.save_type,
                return_savepaths=True,
                save_grid=False
            )
            perform_save_locally_video(
                os.path.join(save_path, "control_hint"),
                control_hint,
                args.target_fps,
                args.save_type,
                save_grid=False
            )
            print("Saved samples to {}. Enjoy.".format(save_path))

            # save video paths
            log_info["video_paths"] += video_paths
            log_info["keyframes_paths"] += keyframes_paths

            # save log info
            with open(os.path.join(save_path, "log_info.json"), "w") as f:
                json.dump(log_info, f, indent=4)

        # back to the original model
        basemodel_idx += 1
        if basemodel_idx < len(basemodel_paths):
            print("--> back to the original model: {}".format(ckpt_path))
            model = model_load_ckpt(model, path=ckpt_path)

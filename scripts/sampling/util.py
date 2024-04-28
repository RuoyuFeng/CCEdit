import os
from itertools import islice

import decord
import cv2
import einops
import imageio
import numpy as np
import PIL.Image as Image
import torch
import torchvision
import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

from sgm.modules.diffusionmodules.sampling import (
    DPMPP2MSampler,
    DPMPP2SAncestralSampler,
    EulerAncestralSampler,
    EulerEDMSampler,
    HeunEDMSampler,
    LinearMultistepSampler,
)
from sgm.modules.encoders.modules import (
    DepthMidasEncoder,
    DepthZoeEncoder,
    LineartEncoder,
    NormalBaeEncoder,
    ScribbleHEDEncoder,
    ScribblePidiNetEncoder,
    SoftEdgeEncoder,
)
from sgm.util import exists, instantiate_from_config, isheatmap


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f"Loaded model config from [{config_path}]")
    return model


def model_load_ckpt(model, path, newbasemodel=False):
    # TODO: how to load ema weights?
    if path.endswith("ckpt") or path.endswith(".pt") or path.endswith(".pth"):
        if "deepspeed" in path:
            sd = torch.load(path, map_location="cpu")
            sd = {k.replace("_forward_module.", ""): v for k, v in sd.items()}
        else:
            # sd = torch.load(path, map_location="cpu")["state_dict"]
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in torch.load(path, map_location="cpu"):
                sd = sd["state_dict"]
    elif path.endswith("safetensors"):
        sd = load_safetensors(path)
    else:
        raise NotImplementedError(f"Unknown checkpoint format: {path}")

    # TODO: (RUOYU) I don't know why need this. We need to refine this for this is really not elegant.
    sd_new = {}
    for k, v in sd.items():
        if k.startswith("conditioner.embedders.") and "first_stage_model" in k:
            loc = k.find("first_stage_model")
            sd_new[k.replace(k[:loc], "")] = v
        else:
            sd_new[k] = v
    sd = sd_new
    del sd_new

    if newbasemodel:
        sd_new = {}
        for k, v in sd.items():
            if "cond_stage_model" in k:
                sd_new[k.replace("cond_stage_model", "conditioner.embedders.0")] = v
                continue
            sd_new[k] = v
        sd = sd_new
        del sd_new

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if newbasemodel:
        unwanted_substrings = ["temporal", "controlnet", "conditioner.embedders.1."]
        missing = [
            each
            for each in missing
            if all(substring not in each for substring in unwanted_substrings)
        ]
    print(
        f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
    )
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
    if len(unexpected) > 0:
        # TODO: notice that some checkpoints has lora parameters (e.g. majicmixRealistic)
        for each in unexpected:
            if each.startswith("lora"):
                print("detected lora parameters, load lora parameters ...", end="\r")
                sd_lora = {}
                for k, v in sd.items():
                    if k.startswith("lora"):
                        sd_lora[k] = v
                        unexpected.remove(k)
                # TODO: alpha?
                sd_lora = convert_load_lora(
                    sd_state_dict=sd, state_dict=sd_lora, alpha=0.8
                )  
                break
        print(f"Unexpected Keys: {unexpected}")

    return model


def convert_load_lora(
    sd_state_dict,
    state_dict,
    LORA_PREFIX_UNET="lora_unet",
    LORA_PREFIX_TEXT_ENCODER="lora_te",
    alpha=0.6,
):
    visited = []

    for key in tqdm.tqdm(state_dict):
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            print("skip: ", key)
            continue

        if "text" in key:
            layer_infos = (
                key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            )
            # curr_layer = pipeline.text_encoder
            if "self_attn" in key:
                layername = "{}.self_attn.{}_proj".format(
                    layer_infos[4], layer_infos[7]
                )
            else:
                layername = "{}.mlp.{}".format(layer_infos[4], layer_infos[-1])
            layername = (
                "cond_stage_model.transformer.text_model.encoder.layers."
                + layername
                + ".weight"
            )
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")

            if "lora_unet_mid_" in key:
                if "_proj_" in key:
                    layername = (
                        "model.diffusion_model.middle_block.1.proj_{}.weight".format(
                            layer_infos[5]
                        )
                    )
                elif "_to_out_" in key:
                    layername = "model.diffusion_model.middle_block.1.transformer_blocks.0.{}.to_out.0.weight".format(
                        layer_infos[7]
                    )
                elif "_ff_net_" in key:
                    layername = "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net"
                    layername = ".".join([layername] + layer_infos[9:]) + ".weight"
                elif "attn1" in key or "attn2" in key:
                    layername = "model.diffusion_model.middle_block.1.transformer_blocks.0.{}.to_{}.weight".format(
                        layer_infos[7], layer_infos[9]
                    )
                else:
                    raise ValueError("Unknown key: ", key)
            else:
                lora_sd_map_in = {
                    "0-0": [1, 1],
                    "0-1": [2, 1],
                    "1-0": [4, 1],
                    "1-1": [5, 1],
                    "2-0": [7, 1],
                    "2-1": [8, 1],
                }
                lora_sd_map_out = {
                    "1-0": [3, 1],
                    "1-1": [4, 1],
                    "1-2": [5, 1],
                    "2-0": [6, 1],
                    "2-1": [7, 1],
                    "2-2": [8, 1],
                    "3-0": [9, 1],
                    "3-1": [10, 1],
                    "3-2": [11, 1],
                }

                if "lora_unet_down_" in key:
                    sd_idxs = lora_sd_map_in[
                        "{}-{}".format(layer_infos[2], layer_infos[4])
                    ]
                    flag_ = "input_blocks"
                elif "lora_unet_up_" in key:
                    sd_idxs = lora_sd_map_out[
                        "{}-{}".format(layer_infos[2], layer_infos[4])
                    ]
                    flag_ = "output_blocks"

                if "_proj_" in key:  # _proj_in and _proj_out
                    layername = "model.diffusion_model.{}.{}.{}.{}_{}.weight".format(
                        flag_, sd_idxs[0], sd_idxs[1], layer_infos[5], layer_infos[6]
                    )
                elif "_to_out_" in key:
                    # model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight
                    layername = "model.diffusion_model.{}.{}.{}.transformer_blocks.{}.{}.to_{}.{}.weight".format(
                        flag_,
                        sd_idxs[0],
                        sd_idxs[1],
                        layer_infos[7],
                        layer_infos[8],
                        layer_infos[10],
                        layer_infos[11],
                    )
                elif "_ff_net_" in key:
                    # model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight
                    layername = "model.diffusion_model.{}.{}.{}.transformer_blocks.{}.ff.net".format(
                        flag_, sd_idxs[0], sd_idxs[1], layer_infos[7]
                    )
                    layername = ".".join([layername] + layer_infos[10:]) + ".weight"
                elif "attn1" in key or "attn2" in key:
                    # model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight
                    layername = "model.diffusion_model.{}.{}.{}.transformer_blocks.{}.{}.to_{}.weight".format(
                        flag_,
                        sd_idxs[0],
                        sd_idxs[1],
                        layer_infos[7],
                        layer_infos[8],
                        layer_infos[10],
                    )
                else:
                    raise ValueError("Unknown key: ", key)
                    # print("Unknown key: {} -> skip".format(key))
                    # continue

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        if "cond_stage_model" in layername:
            layername = layername.replace("cond_stage_model", "conditioner.embedders.0")

        # update weight
        # print('{} -> {}'.format(key, layername))
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = (
                state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            )
            sd_state_dict[layername] += alpha * torch.mm(
                weight_up, weight_down
            ).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            sd_state_dict[layername] += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    print("loading lora done ... ")

    return sd_state_dict


def perform_save_locally_image(save_path, samples):
    assert samples.dim() == 4, "Expected samples to have shape (B, C, H, W)"
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    # samples = embed_watemark(samples)
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:05}.png")
        )
        base_count += 1


def perform_save_locally_video(
    save_path, samples, fps, savetype="gif", return_savepaths=False, save_grid=True,
):
    assert samples.dim() == 5, "Expected samples to have shape (B, C, T, H, W)"
    assert savetype in ["gif", "mp4"]
    os.makedirs(os.path.join(save_path), exist_ok=True)
    os.makedirs(os.path.join(save_path, savetype), exist_ok=True)
    base_count_savetype = len(os.listdir(os.path.join(save_path, savetype)))
    if save_grid:
        os.makedirs(os.path.join(save_path, "grid"), exist_ok=True)
        base_count_grid = len(os.listdir(os.path.join(save_path, "grid")))
    savepaths = []
    for sample in samples:
        t = sample.shape[0]
        sample_grid = einops.rearrange(sample, "c t h w -> t c h w")
        if save_grid:
            torchvision.utils.save_image(
                sample_grid,
                os.path.join(save_path, "grid", f"grid-{base_count_grid:04}.png"),
                nrow=t,
                normalize=False,
                padding=0,
            )

        sample = 255.0 * einops.rearrange(sample.cpu().numpy(), "c t h w -> t h w c")
        sample = sample.astype(np.uint8)
        frames = [each for each in sample]
        if savetype == "gif":
            savepath = os.path.join(
                save_path, "gif", f"animation-{base_count_savetype:04}.gif"
            )
            imageio.mimsave(
                savepath,
                frames,
                format="GIF",
                duration=1 / fps,
                loop=0,
            )
        elif savetype == "mp4":
            savepath = os.path.join(
                    save_path, "mp4", f"animation-{base_count_savetype:04}.mp4"
                )
            # height, width, layers = frames[0].shape
            # size = (width, height)
            # fourcc = cv2.VideoWriter_fourcc(*'avc1')  
            # out = cv2.VideoWriter(savepath, fourcc, fps, size)
            # for frame in frames:
            #     frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            #     out.write(frame_bgr)
            # out.release()
            with imageio.get_writer(savepath, fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)

        else:
            raise ValueError(f"Unknown savetype: {savetype}")
        base_count_savetype += 1
        if save_grid:
            base_count_grid += 1
        savepaths.append(savepath)

    if return_savepaths:
        return savepaths
    else:
        return


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_img(p_cond_img, size: tuple = None):
    """
    Loads an image from the given path and resizes it to the given height and width.
    Converts the image to a tensor and normalizes it to the range [-1, 1]. Shape: (1, 3, H, W)

    Args:
    - p_cond_img (str): path to the image file
    - size (tuple): height and width to resize the image to

    Returns:
    - cond_img (torch.Tensor): tensor of the resized and normalized image.
    """

    cond_img = Image.open(p_cond_img)
    if size:
        assert len(size) == 2, "size should be (H, W)"
        H, W = size
        cond_img = cond_img.resize((W, H), Image.BICUBIC)
    cond_img = np.array(cond_img)
    cond_img = torch.from_numpy(cond_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    cond_img = cond_img * 2.0 - 1.0
    cond_img = torch.clamp(cond_img, -1.0, 1.0)
    return cond_img


def init_sampling(
    sample_steps=50,
    sampler_name="EulerEDMSampler",
    discretization_name="LegacyDDPMDiscretization",
    guider_config_target="sgm.modules.diffusionmodules.guiders.VanillaCFG",
    cfg_scale=7.5,
    img2img_strength=1.0,
):
    assert (
        sample_steps >= 1 and sample_steps <= 1000
    ), "sample_steps must be between 1 and 1000, but got {}".format(sample_steps)
    steps = sample_steps
    assert sampler_name in [
        "EulerEDMSampler",
        "HeunEDMSampler",
        "EulerAncestralSampler",
        "DPMPP2SAncestralSampler",
        "DPMPP2MSampler",
        "LinearMultistepSampler",
    ], "unknown sampler {}".format(sampler_name)
    sampler = sampler_name
    assert discretization_name in [
        "LegacyDDPMDiscretization",
        "EDMDiscretization",
    ], "unknown discretization {}".format(discretization_name)
    discretization = discretization_name

    discretization_config = get_discretization(discretization)

    guider_config = get_guider(
        guider_config_target=guider_config_target, scale=cfg_scale
    )

    sampler = get_sampler(sampler, steps, discretization_config, guider_config)
    if img2img_strength < 1.0:
        from scripts.demo.streamlit_helpers import Img2ImgDiscretizationWrapper

        sampler.discretization = Img2ImgDiscretizationWrapper(
            sampler.discretization, strength=img2img_strength
        )
    return sampler


def get_discretization(discretization):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif discretization == "EDMDiscretization":
        sigma_min = 0.03
        sigma_max = 14.61
        rho = 3.0

        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }

    return discretization_config


def get_guider(
    guider_config_target="sgm.modules.diffusionmodules.guiders.VanillaCFG",
    scale=7.5,
):
    guider = "VanillaCFG"

    if guider == "IdentityGuider":
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        # scale = 7.5
        thresholder = "None"

        if thresholder == "None":
            dyn_thresh_config = {
                "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
            }
        else:
            raise NotImplementedError

        guider_config = {
            "target": guider_config_target,
            "params": {"scale": scale, "dyn_thresh_config": dyn_thresh_config},
        }
    else:
        raise NotImplementedError
    return guider_config


def get_sampler(sampler_name, steps, discretization_config, guider_config):
    if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":
        # default
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = 999.0
        s_noise = 1.0

        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "HeunEDMSampler":
            sampler = HeunEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
    elif (
        sampler_name == "EulerAncestralSampler"
        or sampler_name == "DPMPP2SAncestralSampler"
    ):
        # default
        s_noise = 1.0
        eta = 1.0

        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "LinearMultistepSampler":
        # default
        order = 4
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=order,
            verbose=True,
        )
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def loadmp4_and_convert_to_numpy_cv2(file_path):
   
    """
    Abandoned. This is slow.
    Load an mp4 video file and convert it to a numpy array of frames.

    Args:
        file_path (str): The path to the mp4 video file.

    Returns:
        numpy.ndarray: A numpy array of frames from the video file.
    """
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: Unable to open the file.")
        return None

    frames = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    video_np = np.array(frames)
    video_np = np.flip(video_np, axis=-1)  # BGR to RGB

    return video_np.copy()


def loadmp4_and_convert_to_numpy(file_path):
    """
    Loads an mp4 video file and converts it to a numpy array of frames.

    Args:
        file_path (str): The path to the mp4 video file.

    Returns:
        frames (numpy.ndarray): A numpy array of frames.
    """
    video_reader = decord.VideoReader(file_path, num_threads=0)
    v_len = len(video_reader)
    fps = video_reader.get_avg_fps()
    frames = video_reader.get_batch(list(range(v_len)))
    frames = frames.asnumpy()
    return frames


def load_video(video_path, size: tuple = None, gap: int = 1):
    """
    Load a video from a given path and return a tensor representing the video frames.

    Args:
        size (tuple): The size of the video frames.
        video_path (str): The path to the video file or folder containing the video frames.
        gap (int, optional): The number of frames to skip between each selected frame. Defaults to 1.

    Returns:
        torch.Tensor: A tensor representing the video frames, with shape (T, C, H, W) and values in the range [-1, 1].
    """
    if os.path.isdir(video_path):
        files = sorted(os.listdir(video_path))
        keyfiles = files[::gap]
        frames = [load_img(os.path.join(video_path, kf), size) for kf in keyfiles]
    elif video_path.endswith(".mp4") or video_path.endswith(".gif"):
        if video_path.endswith(".mp4"):
            frames = loadmp4_and_convert_to_numpy(video_path)
        elif video_path.endswith(".gif"):
            frames = imageio.mimread(video_path)
            frames = [np.array(fr) for fr in frames]
            frames = [HWC3(fr) for fr in frames]
            frames = np.stack(frames, axis=0)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        )  # (T, C, H, W)
        frames = frames * 2.0 - 1.0  # range in [-1, 1]
        if size:
            assert len(size) == 2, "size should be (H, W)"
            frames = torch.nn.functional.interpolate(
                frames, size=size, mode="bicubic", align_corners=False
            )
        frames = frames[::gap]  # pick the element every gap frames
        frames = [f.unsqueeze(0) for f in frames]
    else:
        raise ValueError(
            "Unsupported video format. Only support dirctory, .mp4 and .gif."
        )

    return torch.cat(frames, dim=0)  # (T, C, H, W)


def get_keyframes(original_fps, target_fps, allframes, num_keyframes):
    num_allframes = len(allframes)
    gap = np.round(original_fps / target_fps).astype(int)
    assert gap > 0, f"gap {gap} should be positive."
    keyindexs = [i for i in range(0, num_allframes, gap)]
    if len(keyindexs) < num_keyframes:
        print(
            "[WARNING]: not enough keyframes, use linspace instead. "
            f"len(keyindexs): [{len(keyindexs)}] < num_keyframes [{num_keyframes}]"
        )
        keyindexs = np.linspace(0, num_allframes - 1, num_keyframes).astype(int)

    return allframes[keyindexs[:num_keyframes]]


def load_video_keyframes(
    video_path, original_fps, target_fps, num_keyframes, size: tuple = None
):
    """
    Load keyframes from a video file or directory of images.

    Args:
        video_path (str): Path to the video file or directory of images.
        original_fps (int): The original frames per second of the video.
        target_fps (int): The desired frames per second of the output keyframes.
        num_keyframes (int): The number of keyframes to extract.
        size (tuple, optional): The desired size of the output keyframes. Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (T, C, H, W) containing the keyframes.
    """
    if os.path.isdir(video_path):
        files = sorted(os.listdir(video_path))
        num_allframes = len(files)
        gap = np.round(original_fps / target_fps).astype(int)
        assert gap > 0, f"gap {gap} should be positive."
        keyindexs = [i for i in range(0, num_allframes, gap)]
        if len(keyindexs) < num_keyframes:
            print(
                "[WARNING]: not enough keyframes, use linspace instead. "
                f"len(keyindexs): [{len(keyindexs)}] < num_keyframes [{num_keyframes}]"
            )
            keyindexs = np.linspace(0, num_allframes - 1, num_keyframes).astype(int)
        else:
            keyindexs = keyindexs[:num_keyframes]
        keyfiles = [files[i] for i in keyindexs]
        frames = [load_img(os.path.join(video_path, kf), size) for kf in keyfiles]
    elif video_path.endswith(".mp4") or video_path.endswith(".gif"):
        # TODO: not tested yet.
        if video_path.endswith(".mp4"):
            frames = loadmp4_and_convert_to_numpy(video_path)
        elif video_path.endswith(".gif"):
            frames = imageio.mimread(video_path)
            frames = [np.array(fr) for fr in frames]
            frames = [HWC3(fr) for fr in frames]
            frames = np.stack(frames, axis=0)
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        )  # (T, C, H, W)
        num_allframes = frames.shape[0]
        gap = np.round(original_fps / target_fps).astype(int)
        assert gap > 0, f"gap {gap} should be positive."
        keyindexs = [i for i in range(0, num_allframes, gap)]
        if len(keyindexs) < num_keyframes:
            print(
                "[WARNING]: not enough keyframes, use linspace instead. "
                f"len(keyindexs): [{len(keyindexs)}] < num_keyframes [{num_keyframes}]"
            )
            keyindexs = np.linspace(0, num_allframes - 1, num_keyframes).astype(int)
        else:
            keyindexs = keyindexs[:num_keyframes]
        # frames = frames[keyindexs[:num_keyframes]]
        frames = frames[keyindexs]

        frames = frames * 2.0 - 1.0  # range in [-1, 1]
        frames = torch.clamp(frames, -1.0, 1.0)
        if size:
            assert len(size) == 2, "size should be (H, W)"
            frames = torch.nn.functional.interpolate(
                frames, size=size, mode="bicubic", align_corners=False
            )
        # frames = frames[::gap]  # pick the element every gap frames
        frames = [f.unsqueeze(0) for f in frames]
    else:
        raise ValueError(
            "Unsupported video format. Only support dirctory, .mp4 and .gif."
        )

    return torch.cat(frames, dim=0)  # (T, C, H, W)


def setup_controlgenerator(model):
    control_hint_encoder = None
    for embbeder in model.conditioner.embedders:
        if (
            isinstance(embbeder, LineartEncoder)
            or isinstance(embbeder, DepthZoeEncoder)
            or isinstance(embbeder, DepthMidasEncoder)
            or isinstance(embbeder, SoftEdgeEncoder)
            or isinstance(embbeder, NormalBaeEncoder)
            or isinstance(embbeder, ScribbleHEDEncoder)
            or isinstance(embbeder, ScribblePidiNetEncoder)
        ):
            control_hint_encoder = embbeder
            break
    if control_hint_encoder is None:
        raise ValueError("Cannot find LineartEncoder in the embedders.")
    return control_hint_encoder


def load_basemodel_lora(model, basemodel_path="", lora_path=""):
    if basemodel_path:
        print("--> load a new base model from {}".format(basemodel_path))
        model = model_load_ckpt(model, basemodel_path, True)

    if lora_path:
        print("--> load a new LoRA model from {}".format(lora_path))
        sd_state_dict = model.state_dict()

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
            sd_state_dict, lora_state_dict, alpha=1.0
        )  # TODO: alpha
        model.load_state_dict(sd_state_dict)
    return model

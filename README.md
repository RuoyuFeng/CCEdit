# CCEdit
**[CCEdit: Creative and Controllable Video Editing via Diffusion Models](https://arxiv.org/pdf/2309.16496.pdf)**
</br>
Ruoyu Feng,
Wenming Weng,
Yanhui Wang,
Yuhui Yuan,
Jianmin Bao,
Chong Luo,
Zhibo Chen,
Baining Guo

[![arXiv](https://img.shields.io/badge/arXiv-2309.16496-b31b1b.svg)](https://arxiv.org/abs/2309.16496)
[![YoutubeVideo](https://img.shields.io/badge/YoutubeVideo-CCEdit-blue)](https://www.youtube.com/watch?v=UQw4jq-igN4)

<table class="center">
    <tr>
    <td><img src="assets/VideoResults/Interpolation/makeup.gif"></td>
    <td><img src="assets/VideoResults/Interpolation/makeup1-magicReal.gif"></td>
    </tr>
</table>

## Installation
```
# env
conda create -n ccedit python=3.9.17
conda activate ccedit
pip install -r requirements.txt
# pip install -r requirements_pt2.txt
# pip install torch==2.0.1 torchaudio==2.0.2 torchdata==0.6.1 torchmetrics==1.0.0 torchvision==0.15.2
pip install basicsr==1.4.2 wandb loralib av decord timm==0.6.7
pip install moviepy imageio==2.6.0 scikit-image==0.20.0 scipy==1.9.1 diffusers==0.24.0 transformers==4.27.3
pip install accelerate==0.20.3 ujson

git clone https://github.com/lllyasviel/ControlNet-v1-1-nightly src/controlnet11

# download models
download models from https://huggingface.co/RuoyuFeng/CCEdit and put them in ./models

# inference (tv2v)
python scripts/sampling/sampling_tv2v.py   --config_path configs/inference_ccedit/keyframe_no2ndca_depthmidas.yaml   --ckpt_path models/tv2v-no2ndca-depthmidas.ckpt  --H 512 --W 768 --original_fps 18 --target_fps 6 --num_keyframes 17 --batch_size 1 --num_samples 2   --sample_steps 30 --sampler_name DPMPP2SAncestralSampler  --cfg_scale 7.5   --prompt 'a bear is walking.' --video_path assets/Samples/davis/bear   --add_prompt 'Van Gogh style'   --save_path outputs/tv2v/bear-VanGogh   --disable_check_repeat

# inference (tvi2v)
python scripts/sampling/sampling_tv2v_ref.py \
    --seed 201574 \
    --config_path configs/inference_ccedit/keyframe_ref_cp_no2ndca_add_cfca_depthzoe.yaml \
    --ckpt_path models/tvi2v-no2ndca-depthmidas.ckpt \
    --H 512 --W 768 --original_fps 18 --target_fps 6 --num_keyframes 17 --batch_size 1 --num_samples 2 \
    --sample_steps 50 --sampler_name DPMPP2SAncestralSampler --cfg_scale 7 \
    --prompt 'A person walks on the grass, the Milky Way is in the sky, night' \
    --add_prompt 'masterpiece, best quality,' \
    --video_path assets/Samples/tshirtman.mp4 \
    --reference_path assets/Samples/tshirtman-milkyway.png \
    --save_path outputs/tvi2v/tshirtman-MilkyWay \
    --disable_check_repeat \
    --prior_coefficient_x 0.03 \
    --prior_type ref

# train example
python main.py -b configs/example_training/sd_1_5_controlldm-test-ruoyu-tv2v-depthmidas.yaml --wandb False
```

## BibTeX
If you find this work useful for your research, please cite us:

```
@article{feng2023ccedit,
  title={CCEdit: Creative and Controllable Video Editing via Diffusion Models},
  author={Feng, Ruoyu and Weng, Wenming and Wang, Yanhui and Yuan, Yuhui and Bao, Jianmin and Luo, Chong and Chen, Zhibo and Guo, Baining},
  journal={arXiv preprint arXiv:2309.16496},
  year={2023}
}
```

## Conact Us
**Ruoyu Feng**: [ustcfry@mail.ustc.edu.cn](ustcfry@mail.ustc.edu.cn)  


## Acknowledgements
The source videos in this repository come from our own collections and downloads from Pexels. If anyone feels that a particular piece of content is used inappropriately, please feel free to contact me, and I will remove it immediately.

Thanks to model contributers of [CivitAI](https://civitai.com/) and [RunwayML](https://runwayml.com/).

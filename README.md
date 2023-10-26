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

<!-- <p align="center">
  <img src="assets/logo/ccedit.png" width="300">
</p> -->

<table class="center">
    <tr>
    <td><img src="assets/VideoResults/Interpolation/makeup.gif"></td>
    <td><img src="assets/VideoResults/Interpolation/makeup1-magicReal.gif"></td>
    </tr>
</table>

This repository is temporarily used for results display. The code will be released later.

By default, the first video displayed is the original video, and the subsequent ones are the edited results. 

The longer side and the shorter side of the video are 768 and 512 pixels, respectively. The frame number and frame rate is 17 and 6fps, respectively. 



## 🌟 Video Results

### Global Transfer
<div align="center">
    <img src="assets/Gallery/Human/smile-revAnimatedlineart-cat.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) A young and beautiful girl smiles to the camera, anime style.</p>

<div align="center">
    <img src="assets/VideoResults/GlobalTransfer/City3-cyberpunk.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/30240">ToonYou</a>) City night, cyberpunk style.</p>

<div align="center">
    <img src="assets/VideoResults/GlobalTransfer/dubinrunning-mechaAnimal.gif">
    </video>
</div>

<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>, <a href="https://civitai.com/models/63999?modelVersionId=68589">kMechAnimal</a>) (w/o reference frame) A mechanical Doberman is running. </p>

<div align="center">
    <img src="assets/Gallery/Animal/revAnimated_kMechaAnimal.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>, <a href="https://civitai.com/models/63999?modelVersionId=68589">kMechAnimal</a>) A mechanical bear is running. </p>

<div align="center">
    <img src="assets/VideoResults/GlobalTransfer/motorcycle-paladin.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) A paladin drives a motorcycle, fire on the road.</p>

<table class="center">
    <tr>
    <td><img src="assets/VideoResults/GlobalTransfer/suitguy-mecha.gif"></td>
    <td><img src="assets/VideoResults/GlobalTransfer/hoodguy-magicianblueeye.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/69438/mechamix">MechaMix</a>) Portrait shot of robot, mechanical, sophisticated ancient Egypt style filigree inlays, cyberpunk, vibrant color, half body, dark vibes, volumetric light, dramatic background.</p>
<p style="margin-left: 0em; margin-top: -1em">(Model: <a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) A magician in hood, blue eye, blue flame.</p>


<table class="center">
    <tr>
    <td><img src="assets/VideoResults/GlobalTransfer/1-magicreal2.gif"></td>
    <td><img src="assets/VideoResults/GlobalTransfer/Hat-counterfeitsoftedge2.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/43331/majicmix-realistic">majicMIX realistic</a>) A young and beautiful girl.</p>
<p style="margin-left: 0em; margin-top: -1em">(Model: <a href="https://civitai.com/models/4468?modelVersionId=57618">Counterfeit-V3.0</a>) A cute girl with a hat.</p>



### Foreground Editing
<div align="center">
    <img src="assets/VideoResults/Foreground/corgi-fat.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/108289?modelVersionId=116540">hellofantasytime</a>, <a href="https://civitai.com/models/110738?modelVersionId=119401">fat animal</a>) A cute corgi stick out tongue.</p>

<div align="center">
    <img src="assets/VideoResults/Foreground/weilai1-mecha.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/110768?modelVersionId=119447">hellomecha</a>, <a href="https://civitai.com/models/130742?modelVersionId=143505">Building_block_world</a>) A Lego brick -style car stops on the road. BJ_Lego bricks, no_humans, ground_vehicle, motor_vehicle, science_fiction, vehicle_focus, cinematic lighting, strong contrast, high level of detail.</p>

<div align="center">
    <img src="assets/VideoResults/Foreground/tiger-anime.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/30240">ToonYou</a>) A tiger is walking, anime style.</p>

<div align="center">
    <img src="assets/VideoResults/Foreground/womanhair-anime.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/4468?modelVersionId=57618">Counterfeit-V3.0</a>) A young girl, anime style.</p>


<div align="center">
    <img src="assets/VideoResults/Foreground/bomei_anime.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/4468?modelVersionId=57618">Counterfeit-V3.0</a>) A cute dog ran towards the camera.</p>


### Background Editing
<div align="center">
    <img src="assets/VideoResults/Background/runningguykuanping-sunset.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/30240">ToonYou</a>) A man is running on the beach, sunset.</p>

<div align="center">
    <img src="assets/VideoResults/Background/tshirtman-MilkyWay.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) A person walks in the field. The Milky Way is in the sky, at night. </p>

<div align="center">
    <img src="assets/VideoResults/Background/yoga2-snow.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/30240">ToonYou</a>) A woman is doing yoga, in winter, snow.</p>

<table class="center">
    <tr>
    <td><img src="assets/VideoResults/Background/womanback-sunsetanime.gif"></td>
    <td><img src="assets/VideoResults/Background/womandrink-sping.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/30240">ToonYou</a>) a woman is walking on the country road, sunset, back to the camera.</p>
<p style="margin-left: 0em; margin-top: -1em">(Model: <a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">SD-v1.5</a>) A woman is drinking wine in a spring field.</p>

<table class="center">
    <tr>
    <td><img src="assets/Gallery/Human/suitguyback_technique.gif"></td>
    <td><img src="assets/Gallery/Human/guyhorse_magicword.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) A man in a suit walks into a technological city, feeling futuristic and cinematic.</p>
<p style="margin-left: 0em; margin-top: -1em">(Model: <a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) A man with black suit and a black horse walk in the wood.</p>


## 🌟 Features

### Different Styles
<table class="center">
    <tr>
    <td><img src="assets/Gallery/Landscape/City3.gif"></td>
    <td><img src="assets/Gallery/Landscape/City3-anime.gif"></td>
    <td><img src="assets/Gallery/Landscape/City3-cyberpunk.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/30240">ToonYou</a>) City, anime style. </p>
<p style="margin-left: 0em; margin-top: -1em">(Model: <a href="https://civitai.com/models/30240">ToonYou</a>) City at night, cyberpunk stile. </p>

<table class="center">
    <tr>
    <td><img src="assets/Gallery/Objects/aircraftcarrier.gif"></td>
    <td><img src="assets/Gallery/Objects/aircraftcarrier-lego.gif"></td>
    <td><img src="assets/Gallery/Objects/aircraftcarrier-spaceship.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/110768?modelVersionId=119447">hellomecha</a>, <a href="https://civitai.com/models/130742?modelVersionId=143505">Building_block_world</a>) A LEGO-style aircraft carrier.</p>
<p style="margin-left: 0em; margin-top: -1em">(Model: <a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) Spaceship flys in the sky.</p>


<table class="center">
    <tr>
    <td><img src="assets/Gallery/Human/HatLonger.gif"></td>
    <td><img src="assets/Gallery/Human/Hat-aniflatmixdepth.gif"></td>
    <td><img src="assets/Gallery/Human/Hat_majicmixRealisticbetterV2V25.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">(Model: <a href="https://civitai.com/models/4468?modelVersionId=57618">Counterfeit-V3.0</a>) A girl, anime style. </p>
<p style="margin-left: 0em; margin-top: -1em">(Model: <a href="https://civitai.com/models/43331/majicmix-realistic">majicMIX realistic</a>) A girl.</p>


<table class="center">
    <tr>
    <td><img src="assets/Gallery/Human/1.gif"></td>
    <td><img src="assets/Gallery/Human/1-magicreal2.gif"></td>
    <td><img src="assets/Gallery/Human/1-lineart.gif"></td>
    <td><img src="assets/Gallery/Human/1-toonyousoftedge.gif"></td>
    <td><img src="assets/Gallery/Human/1-toonyoudepth.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/43331/majicmix-realistic">majicMIX realistic</a>)
A girl.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>) 
A girl, anime style.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>) 
A girl, anime style.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>) 
A girl, anime style.
</p>


### Different Granularities
<table class="center">
    <tr>
    <td><img src="assets/Gallery/Human/manwaterfall2.gif"></td>
    <td><img src="assets/Gallery/Human/manwaterfall2-lineart.gif"></td>
    <td><img src="assets/Gallery/Human/manwaterfall2-softedge.gif"></td>
    </tr>
</table>

<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/36520?modelVersionId=76907">GhostMix</a> , <a href="https://civitai.com/models/63347/ink-splash/">泼墨 ink splash</a>) (w/o reference frame) A man holding a sword stands in front of a waterfall, cold face, ink splash style.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/36520?modelVersionId=76907">GhostMix</a> , <a href="https://civitai.com/models/63347/ink-splash/">泼墨 ink splash</a>) (w/o reference frame) A man holding a sword stands in front of a waterfall, cold face, ink splash style.
</p>

<table class="center">
    <tr>
    <td><img src="assets/Gallery/Human/manwaterfall1.gif"></td>
    <td><img src="assets/Gallery/Human/manwaterfall1-lineart.gif"></td>
    <td><img src="assets/Gallery/Human/manwaterfall1-softedge.gif"></td>
    <td><img src="assets/Gallery/Human/manwaterfall1-inklineart.gif"></td>
    <td><img src="assets/Gallery/Human/manwaterfall1-inkdepthpose4.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/36520?modelVersionId=76907">GhostMix</a> , <a href="https://civitai.com/models/63347/ink-splash/">泼墨 ink splash</a>) (w/o reference frame) A man splits the water surface with a sword, beneath a waterfall, ink splash style.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/36520?modelVersionId=76907">GhostMix</a> , <a href="https://civitai.com/models/63347/ink-splash/">泼墨 ink splash</a>) (w/o reference frame) A man splits the water surface with a sword, beneath a waterfall, ink splash style.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/36520?modelVersionId=76907">GhostMix</a> , <a href="https://civitai.com/models/63347/ink-splash/">泼墨 ink splash</a>) A man splits the water surface with a sword, beneath a waterfall, ink splash style.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/36520?modelVersionId=76907">GhostMix</a> , <a href="https://civitai.com/models/63347/ink-splash/">泼墨 ink splash</a>) A man splits the water surface with a sword, beneath a waterfall, ink splash style.
</p>

### Different Content

<table class="center">
    <tr>
    <td><img src="assets/Gallery/Animal/tiger.gif"></td>
    <td><img src="assets/Gallery/Animal/tiger-tiger.gif"></td>
    <td><img src="assets/Gallery/Animal/tiger-bear.gif"></td>
    <td><img src="assets/Gallery/Animal/tiger-panda2.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>)
A tiger is walking, anime style.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>)
A bear is walking, anime style.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>)
A panda is walking, anime style.
</p>

## 🌟 Gallery
Here are some more samples of our results. 

<div align="center">
    <img src="assets/Gallery/Human/flower_revAnimated.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) (w/o reference frame) A beautiful woman sits in grass and smiles, flowers in background.
</p>

<div align="center">
    <img src="assets/Gallery/Human/Yoga-animestyle.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>) A woman is doing yoga, anime style.
</p>

<div align="center">
    <img src="assets/Gallery/Landscape/ChongqingNight1-cyberpunk2.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>) City night, cyberpunk style.
</p>

<div align="center">
    <img src="assets/Gallery/Landscape/city-anime.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>) City, anime style.
</p>

<div align="center">
    <img src="assets/Gallery/Objects/aircraftcarrier2-spaceship.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) A spaceship flying in the space, galaxy background, ultra detailed, Hyperrealistic, sharp focus, UHD, octane render.
</p>

<div align="center">
    <img src="assets/Gallery/Objects/yacht7-spaceship2.gif">
    </video>
</div>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) A spaceship flying over the city.
</p>

<table class="center">
    <tr>
    <td><img src="assets/Gallery/Objects/yacht3.gif"></td>
    <!-- <td><img src="assets/Gallery/Objects/yacht3-spaceship.gif"></td> -->
    <!-- <td><img src="assets/Gallery/Objects/yacht3-spaceship2.gif"></td> -->
    <td><img src="assets/Gallery/Objects/yacht3-spaceship3.gif"></td>
    <td><img src="assets/Gallery/Objects/yacht3-spaceship4.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) A spaceship flying over the city.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/7371/rev-animated">ReV Animated</a>) A spaceship flying over the city.
</p>

<table class="center">
    <tr>
    <td><img src="assets/Gallery/Landscape/QingguiNight.gif"></td>
    <td><img src="assets/Gallery/Landscape/QingguiNight-anime2.gif"></td>
    <td><img src="assets/Gallery/Landscape/QingguiNight-anime3.gif"></td>
    <!-- <td><img src="assets/Gallery/Landscape/QingguiNight-anime4.gif"></td> -->
    <td><img src="assets/Gallery/Landscape/QingguiNight-animeflat.gif"></td>
    </tr>
</table>
<p style="margin-left: 0em; margin-top: 1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>) 
A light rail passes by in the city at night, anime style.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/30240">ToonYou</a>) 
A light rail passes by in the city at night, anime style, high contrast.
</p>
<p style="margin-left: 0em; margin-top: -1em">
(Model: 
<a href="https://civitai.com/models/24387?modelVersionId=40816">Aniflatmix - Anime Flat Color Style Mix (平涂り風/平涂风)</a>) 
A light rail passes by in the city at night, anime style.
</p>


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

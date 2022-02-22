[简体中文](README.md) | English

# PaddleVideo

​ 💖 **Welcome to scan the code and join the group discussion** 💖

<div align="center">
  <img src="docs/images/user_group.png" width=250/></div>

- Scan the QR code below with your Wechat and reply "video", you can access to official technical exchange group. Look forward to your participation.

## Introduction

![python version](https://img.shields.io/badge/python-3.7+-orange.svg) ![paddle version](https://img.shields.io/badge/PaddlePaddle-2.0-blue )


PaddleVideo is a toolset for video tasks prepared for the industry and academia. This repository provides examples and best practice guildelines for exploring deep learning algorithm in the scene of video area.

<div align="center">
  <img src="docs/images/home.gif" width="450px"/><br>
</div>


## Model and Applications

### Model zoo

- Please refer to [Installation guide](docs/zh-CN/install.md) and [Usage doc](docs/zh-CN/usage.md) before using the model zoo.

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">Action recognition method</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/recognition/pp-tsm.md">PP-TSM</a> (PP series)</td>
    <td><a href="./docs/en/model_zoo/recognition/pp-tsn.md">PP-TSN</a> (PP series)</td>
    <td><a href="./docs/en/model_zoo/recognition/pp-timesformer.md">PP-TimeSformer</a> (PP series)</td>
    <td><a href="./docs/en/model_zoo/recognition/tsn.md">TSN</a> (2D’)</td>
    <td><a href="./docs/en/model_zoo/recognition/tsm.md">TSM</a> (2D')</td>
  <tr>
    <td><a href="./docs/en/model_zoo/recognition/slowfast.md">SlowFast</a> (3D’)</td>
    <td><a href="./docs/en/model_zoo/recognition/timesformer.md">TimeSformer</a> (Transformer')</td>
    <td><a href="./docs/en/model_zoo/recognition/videoswin.md">VideoSwin</a> (Transformer’)</td>
    <td><a href="./docs/en/model_zoo/recognition/attention_lstm.md">AttentionLSTM</a> (RNN')</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Skeleton based action recognition</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/recognition/stgcn.md">ST-GCN</a> (Custom’)</td>
    <td><a href="./docs/en/model_zoo/recognition/agcn.md">AGCN</a> (Adaptive')</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Sequence action detection method</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/localization/bmn.md">BMN</a> (One-stage')</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Spatio-temporal motion detection method</td>
  </tr>
  <tr>
    <td><a href="docs/en/model_zoo/detection/SlowFast_FasterRCNN_en.md">SlowFast+Fast R-CNN</a>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Multimodal</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/multimodal/actbert.md">ActBERT</a> (Learning')</td>
    <td><a href="./applications/T2VLAD/README.md">T2VLAD</a> (Retrieval')</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Video target segmentation</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/segmentation/cfbi.md">CFBI</a> (Semi')</td>
    <td><a href="./applications/EIVideo/EIVideo/docs/en/manet.md">MA-Net</a> (Supervised')</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Monocular depth estimation</td>
  </tr>
  <tr>
    <td><a href="./docs/en/model_zoo/estimation/adds.md">ADDS</a> (Unsupervised‘)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>


### Dataset

<table>
  <tbody><tr>
    <td colspan="4">Action Recognition</td>
  </tr>
  <tr>
    <td><a href="docs/en/dataset/k400.md">Kinetics-400</a> (<a href="https://deepmind.com/research/open-source/kinetics/" rel="nofollow">Homepage</a>) (CVPR'2017)</td>
    <td><a href="docs/en/dataset/ucf101.md">UCF101</a> (<a href="https://www.crcv.ucf.edu/research/data-sets/ucf101/" rel="nofollow">Homepage</a>) (CRCV-IR-12-01)</td>
    <td><a href="docs/en/dataset/ActivityNet.md">ActivityNet</a> (<a href="http://activity-net.org/" rel="nofollow">Homepage</a>) (CVPR'2015)</td>
    <td><a href="docs/en/dataset/youtube8m.md">YouTube-8M</a> (<a href="https://research.google.com/youtube8m/" rel="nofollow">Homepage</a>) (CVPR'2017)</td>
  </tr>
  <tr>
    <td colspan="4">Action Localization</td>
  </tr>
  <tr>
    <td><a href="docs/en/dataset/ActivityNet.md">ActivityNet</a> (<a href="http://activity-net.org/" rel="nofollow">Homepage</a>) (CVPR'2015)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4">Spatio-Temporal Action Detection</td>
  </tr>
  <tr>
    <td><a href="docs/en/dataset/AVA.md">AVA</a> (<a href="https://research.google.com/ava/index.html" rel="nofollow">Homepage</a>) (CVPR'2018)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4">Skeleton-based Action Recognition</td>
  </tr>
  <tr>
    <td><a href="docs/en/dataset/ntu-rgbd.md">NTURGB+D</a> (<a href="https://rose1.ntu.edu.sg/dataset/actionRecognition/" rel="nofollow">Homepage</a>) (IEEE CS'2016)</td>
    <td><a href="docs/en/dataset/fsd.md">FSD</a> (<a href="https://aistudio.baidu.com/aistudio/competition/detail/115/0/introduction" rel="nofollow">Homepage</a>)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4">Depth Estimation</td>
  </tr>
  <tr>
    <td><a href="docs/en/dataset/Oxford_RobotCar.md">Oxford-RobotCar</a> (<a href="https://robotcar-dataset.robots.ox.ac.uk/" rel="nofollow">Homepage</a>) (IJRR'2017)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4">Text-Video Retrieval</td>
  </tr>
  <tr>
    <td><a href="docs/zh-CN/dataset/msrvtt.md">MSR-VTT</a> (<a href="https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/" rel="nofollow">Homepage</a>) (CVPR'2016)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4">Text-Video Pretrained Model</td>
  </tr>
  <tr>
    <td><a href="docs/zh-CN/dataset/howto100m.md">HowTo100M</a> (<a href="https://www.di.ens.fr/willow/research/howto100m/" rel="nofollow">Homepage</a>) (ICCV'2019)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
</table>


### Applications

| Applications | Descriptions |
| :--------------- | :------------ |
| [FootballAction]() | Football action detection solution|
| [BasketballAction](applications/BasketballAction) | Basketball action detection solution |
| [TableTennis](applications/ableTennis) | Table tennis action recognition solution|
| [FigureSkating](applications/FigureSkating) | Figure skating action recognition solution|
| [VideoTag](applications/VideoTag) | 3000-category large-scale video classification solution |
| [MultimodalVideoTag](applications/MultimodalVideoTag) | Multimodal video classification solution|
| [VideoQualityAssessment](applications/VideoQualityAssessment) | Video quality assessment solution|
| [PP-Care](applications/PP-Care) | 3DMRI medical image recognition solution |
| [EIVideo](applications/EIVideo) | Interactive video segmentation tool|
| [Anti-UAV](applications/Anti-UAV) |UAV detection solution|
| [AbnormalActionDetection](applications/AbnormalActionDetection) |Abnormal action detection solution|


## Documentation tutorial
- AI-Studio Tutorial
    - [[Official] Paddle2.1 realizes video understanding optimization model -- PP-TSM](https://aistudio.baidu.com/aistudio/projectdetail/3399656?contributionType=1)
    - [[Official] Paddle2.1 realizes video understanding optimization model -- PP-TSN](https://aistudio.baidu.com/aistudio/projectdetail/2879980?contributionType=1)
    - [[Official] Paddle 2.1 realizes the classic model of video understanding - TSN](https://aistudio.baidu.com/aistudio/projectdetail/2250682)
    - [[Official] Paddle 2.1 realizes the classic model of video understanding - TSM](https://aistudio.baidu.com/aistudio/projectdetail/2310889)
    - [BMN video action positioning](https://aistudio.baidu.com/aistudio/projectdetail/2250674)
    - [ST-GCN Tutorial for Figure Skate Skeleton Point Action Recognition](https://aistudio.baidu.com/aistudio/projectdetail/2417717)
    - [[Practice]video understanding transformer model TimeSformer](https://aistudio.baidu.com/aistudio/projectdetail/3413254?contributionType=1)
- Contribute code
    - [How to add a new algorithm](./docs/zh-CN/contribute/add_new_algorithm.md)
    - [Configuration system design analysis](./docs/en/tutorials/config.md)
    - [How to mention PR](./docs/zh-CN/contribute/how_to_contribute.md)


## Competition

- [Figure skating action recoginition using skeleton based on PaddlePaddle](https://aistudio.baidu.com/aistudio/competition/detail/115/0/introduction), [AI Studio projects](https://aistudio.baidu.com/aistudio/projectdetail/2417717), [video course](https://www.bilibili.com/video/BV1w3411172G)
- [Table tennis action proposal localization based on PaddlePaddle](https://aistudio.baidu.com/aistudio/competition/detail/127/0/introduction)
- [CCKS 2021: Knowledge Augmented Video Semantic Understanding](https://www.biendata.xyz/competition/ccks_2021_videounde)

## License

PaddleVideo is released under the [Apache 2.0 license](LICENSE).


## Thanks
- Many thanks to [mohui37](https://github.com/mohui37)、[zephyr-fun](https://github.com/zephyr-fun)、[voipchina](https://github.com/voipchina) for contributing the code for prediction.

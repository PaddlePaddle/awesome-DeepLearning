# SlimX系列小模型

PaddleSlim模型压缩工具在人脸识别,OCR,通用任务分类任务，检测任务等多个任务上都发布了SlimX系列小模型:

- `SlimMobileNet系列`
- `SlimFaceNet系列`

## SlimMobileNet系列指标

SlimMobileNet基于百度自研的[GP-NAS论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_GP-NAS_Gaussian_Process_Based_Neural_Architecture_Search_CVPR_2020_paper.pdf)（CVPR2020）AutoDL技术以及自研的蒸馏技术得到。

相比于MobileNetV3, SlimMobileNet_V1在精度提升1.7个点的情况下Flops可以压缩138%。
由于精度比MobileNetV3高出了1.7个点，SlimMobileNet_V1量化后精度仍然高于MobileNetV3。量化后SlimMobileNet_V1可以在精度高于MobileNetV3的情况下Flops压缩552%。SlimMobileNet_V4_x1_1为业界首次发布的Flops 300M以下，ImagenetNet精度超过80%的分类小模型。

|Method|Flops(M)|Top1 Acc|
|------|-----|-----|
|MobileNetV3_large_x1_0|225|75.2|
|MobileNetV3_large_x1_25|357|76.6|
|GhostNet_x1_3|220|75.7|
|SlimMobileNet_V1|163|76.9|
|SlimMobileNet_V4_x1_1|296|80.1|
|SlimMobileNet_V5|390|80.4|

## [SlimFaceNet](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/slimfacenet/README.md)系列指标

SlimFaceNet同样是基于百度自研的GP-NAS AutoDL技术以及百度自研的自监督超网络训练算法得到。相比于MobileNetV2，SlimFaceNet_A_x0_60 flops压缩216%，在RK3288上加速428%。基于PaddleSlim的离线量化功能还可以进一步压缩模型，相比于MobileNetV2，SlimFaceNet_A_x0_60_quant flops可以压缩865%，在RK3288硬件上可以加速643%。为了对齐论文，LFW指标为112x96输入下的结果；结合业务场景，Flops和speed为112x112输入下的结果，延时为RK3288上的延时。

|Method|LFW|Flops|speed|
|------|-----|-----|-----|
|MobileNetV2|98.58%|277M|270ms|
|MobileFaceNet|99.18%|224M|102ms|
|SlimFaceNet_A_x0_60|99.21%|128M|63ms|
|SlimFaceNet_B_x0_75|99.22%|151M|70ms|
|SlimFaceNet_A_x0_60_quant|99.17%|32M|42ms|
|SlimFaceNet_B_x0_75_quant|99.21%|38M|45ms|

## 业界领先的AutoDL技术

GP-NAS从贝叶斯角度来建模NAS，并为不同的搜索空间设计了定制化的高斯过程均值函数和核函数。 具体来说，基于GP-NAS的超参数，我们有能力高效率的预测搜索空间中任意模型结构的性能。 从而，模型结构自动搜索问题就被转
换为GP-NAS高斯过程的超参数估计问题。接下来，通过互信息最大化采样算法，我们可以有效地对模型结构进行采样。 因此，根据采样网络的性能，我们可以有效的逐步更新GP-NAS超参数的后验分布。基于估计出的GP-NAS超参数，
我们可以预测出满足特定延时约束的最优的模型结构，更详细的技术细节请参考GP-NAS论文。

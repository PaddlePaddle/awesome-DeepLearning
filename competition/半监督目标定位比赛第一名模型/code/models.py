from backcbone.Res2Net import Res2Net200_vd_26w_4s_ssld
from backcbone.ResNeXt import ResNeXt101_32x16d_wsl
from backcbone.SwinTransformer import SwinTransformer_large_patch4_window12_384
from Decoder.ACFFNet import ACFFModel
from Decoder.FMFNet import FMFModel
from Decoder.ACFFViT import ACFFViTModel


def Res2NetandACFFNet():
    return ACFFModel(backbone=Res2Net200_vd_26w_4s_ssld(pretrained=False))


def Res2NetandFMFNet():
    return FMFModel(backbone=Res2Net200_vd_26w_4s_ssld(pretrained=False))


def ResNeXtandACFFNet():
    return ACFFModel(backbone=ResNeXt101_32x16d_wsl(pretrained=False))


def SwinTandACFFNet():
    return ACFFViTModel(backbone=SwinTransformer_large_patch4_window12_384(pretrained=False))

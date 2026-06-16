import torch
from thop import profile, clever_format
from torchsummary import summary



def get_model(name: str, in_channels: int, num_classes: int, dataset_name: str = None):
    if name == "CVCNN":
        from .CVCNN import CVCNN
        num_blocks = 6 if dataset_name == "WiSig_ManyRx" else 9
        return CVCNN(
            in_channels=in_channels,
            channels=64,
            num_classes=num_classes,
            num_blocks=num_blocks
        )

    elif name == "LightCNN":
        from .LightCNN import LightCNN
        return LightCNN(
            in_channels=2,
            channels=64,
            num_classes=num_classes,
            dropout_p=0.5
        )

    elif name == "MSCAN":
        from .MACNNmodel import MACNN
        return MACNN(
            in_channels=in_channels,
            channels=64,
            num_classes=num_classes,
            block_num=[2, 2, 2]
        )

    elif name == "ResNet" or name == "ResNet18":
        from .ResNet_model import resnet18
        return resnet18(num_class=num_classes, in_channels=in_channels)


    else:
        raise ValueError(f"未知模型: {name}")
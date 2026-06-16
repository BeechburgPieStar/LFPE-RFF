import torch
from .modules.complexcnn import ComplexConv
from torch import nn
import torch.nn.functional as F


class ComplexConvBlock(nn.Module):
    def __init__(self, in_complex_channels: int, out_complex_channels: int,
                 kernel_size: int = 3, pool_kernel: int = 2, pool_stride: int = 2):     ##构造函数
        super().__init__()
        self.conv = ComplexConv(in_channels=in_complex_channels,
                                out_channels=out_complex_channels,
                                kernel_size=kernel_size)                  ##卷积层
        self.bn = nn.BatchNorm1d(num_features=out_complex_channels * 2)   ##批归一化层
        self.act = nn.ReLU(inplace=True)                                  ##激活函数（负值变为0）
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)   ##池化层，下采样

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class CVCNN(nn.Module):
    def __init__(self, in_channels=2, channels=64, num_classes=121, num_blocks=9):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels 必须是偶数"
        in_complex = in_channels // 2

        blocks = []    ##blocks列表会包含多个ComplexConvBlock层
        blocks.append(ComplexConvBlock(in_complex_channels=in_complex,
                                       out_complex_channels=channels))      ##把输入从 in_complex 个复通道（例如 1 组 IQ）映射到 channels 个复通道
                                                                            ##（默认 64 组 IQ 特征 ⇒ 128 条实通道）
        for _ in range(num_blocks - 1):
            blocks.append(ComplexConvBlock(in_complex_channels=channels,
                                           out_complex_channels=channels))          ##复通道数保持为 64（恒宽）

        self.backbone = nn.Sequential(*blocks)       ##将blocks列表中的所有层（在本例中是多个 ComplexConvBlock 层）按顺序连接成一
                                                    # 个 Sequential 容器，并将这个容器赋值给 self.backbone

        self.flatten = nn.Flatten()           ##把 (N, C, L) 拉平成 (N, C*L)
        self.fc1 = nn.LazyLinear(1024)        ##fc1 是第一个全连接层，它的输入是展平后的张量 x，形状为 (N, C*L)，
                                              # 输出是一个具有 1024 个神经元的向量
        self.fc2 = nn.LazyLinear(num_classes)            #分类，接受fc1的输出，然后输出（N，num_classes）

    def forward(self, x):
        x = self.backbone(x)                ##构造神经网络
        x = self.flatten(x)
        x = self.fc1(x)
        embed = F.relu(x)              ##嵌入向量 embed（大小 1024），这通常是你想要用于度量、可视化或能量计算的“表征”
        logits = self.fc2(embed)         ##分类头输出 logits
        return embed, logits


from thop import profile, clever_format
from torchsummary import summary


def main():
    batch_size = 1          ##fc1输出张量（batch_size，1024）
    in_channels = 2   # 例如复数1路 => 实/虚两路拼成2
    seq_len = 6000
    num_classes = 10

    device = "cpu"
    model = CVCNN(in_channels=in_channels, channels=64,
                  num_classes=num_classes, num_blocks=9).to(device).eval()

    summary(model, input_size=(in_channels, seq_len), device=str(device))

    x = torch.randn(batch_size, in_channels, seq_len, device=device)

    with torch.no_grad():
        _ = model(x)  # 简单前向，确认能跑

    macs, params = profile(model, inputs=(x,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")

    print("Params:", params)
    print("MACs:  ", macs)

if __name__ == "__main__":
    main()
import torch.nn as nn


class ConvModule2D(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 kernel_size,
                 dilation,
                 stride):
        super().__init__()
        # for stride 1
        h_pad = int(dilation[0] * (kernel_size[0] - 1) // 2)
        w_pad = int(dilation[1] * (kernel_size[1] - 1) // 2)
        self._conv_module = nn.Conv2d(in_channels=in_c,
                                      out_channels=out_c,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=(h_pad, w_pad),
                                      padding_mode=r"zeros",
                                      dilation=(dilation[0], dilation[1]),
                                      groups=1,
                                      bias=False)
        nn.init.kaiming_normal_(self._conv_module.weight)

    def forward(self, x):
        out = self._conv_module(x)
        return out


class LinearModule(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim):
        super().__init__()
        self._linear_layer = nn.Linear(in_features=in_dim,
                                       out_features=out_dim,
                                       bias=False)
        nn.init.normal_(self._linear_layer.weight, 0.0, 0.01)

    def forward(self, x):
        out = self._linear_layer(x)
        return out

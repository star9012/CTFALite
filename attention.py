import math
import torch
import torch.nn as nn
from layers import ConvModule2D


class SEModule(nn.Module):
    def __init__(self,
                 in_c,
                 r,
                 ac_fn):
        super(SEModule, self).__init__()
        self._dim = int(in_c // r)
        self._se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule2D(in_c=in_c,
                         out_c=self._dim,
                         kernel_size=(1, 1),
                         stride=(1, 1),
                         dilation=(1, 1)),
            ac_fn,
            ConvModule2D(in_c=self._dim,
                         out_c=in_c,
                         kernel_size=(1, 1),
                         stride=(1, 1),
                         dilation=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, batch):
        att = self._se(batch)
        out = batch * att
        return out


class ECA(nn.Module):
    def __init__(self,
                 in_c):
        super().__init__()
        k1 = int(abs(math.log2(in_c) / 2 + 0.5))
        k2 = k1 if k1 % 2 else k1 + 1
        self._gap = nn.AdaptiveAvgPool2d(1)
        self._conv = ConvModule2D(in_c=1,
                                  out_c=1,
                                  kernel_size=(k2, 1),
                                  stride=(1, 1),
                                  dilation=(1, 1))
        self._sigmoid = nn.Sigmoid()

    def forward(self, feature_maps):
        gap_out = self._gap(feature_maps)
        eca_weights = self._sigmoid(self._conv(gap_out.transpose(1, 2)))
        enhanced_out = torch.multiply(feature_maps, eca_weights.transpose(1, 2))
        return enhanced_out


class TFCBAM(nn.Module):
    def __init__(self, in_c, r, ac_fn):
        super().__init__()
        self._ac_fn = ac_fn
        self._dim = int(in_c // r)
        self._sigmoid = nn.Sigmoid()
        self._gap = nn.AdaptiveAvgPool2d((1, 1))
        self._gmp = nn.AdaptiveMaxPool2d((1, 1))
        self._conv1 = ConvModule2D(in_c=in_c,
                                   out_c=self._dim,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   dilation=(1, 1))
        self._conv2 = ConvModule2D(in_c=self._dim,
                                   out_c=in_c,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   dilation=(1, 1))
        self._conv3 = ConvModule2D(in_c=2,
                                   out_c=in_c,
                                   kernel_size=(7, 1),
                                   stride=(1, 1),
                                   dilation=(1, 1))
        self._conv4 = ConvModule2D(in_c=2,
                                   out_c=in_c,
                                   kernel_size=(1, 7),
                                   stride=(1, 1),
                                   dilation=(1, 1))

    def forward(self, feature_maps):
        batch_size, channel, frequency, time = feature_maps.size()
        gap = self._gap(feature_maps)
        gmp = self._gmp(feature_maps)
        gap_conv = self._ac_fn(self._conv1(gap))
        gmp_conv = self._ac_fn(self._conv1(gmp))
        conv_sum = self._conv2(gap_conv) + self._conv2(gmp_conv)
        c_att = self._sigmoid(conv_sum)
        c_feature_maps = torch.multiply(feature_maps, c_att)
        f = torch.mean(c_feature_maps, dim=-1)
        t = torch.mean(c_feature_maps, dim=-2)
        # f: (batch_size, channel, frequency), t: (batch_size, channel, time)
        f = f.transpose(1, 2)
        f = f.view(batch_size, frequency, channel, 1)
        f_avg = self._gap(f)
        f_avg = f_avg.view(batch_size, 1, frequency, 1)
        f_max = self._gmp(f)
        f_max = f_max.view(batch_size, 1, frequency, 1)
        f_cat = torch.cat((f_avg, f_max), dim=1)
        f_att = self._sigmoid(self._conv3(f_cat))

        t = t.transpose(1, 2)
        t = t.view(batch_size, time, channel, 1)
        t_avg = self._gap(t)
        t_avg = t_avg.view(batch_size, 1, 1, time)
        t_max = self._gmp(t)
        t_max = t_max.view(batch_size, 1, 1, time)
        t_cat = torch.cat((t_avg, t_max), dim=1)
        t_att = self._sigmoid(self._conv4(t_cat))

        output = 0.5 * torch.multiply(c_feature_maps, f_att) + 0.5 * torch.multiply(c_feature_maps, t_att)
        return output


class CA(nn.Module):
    def __init__(self,
                 in_c,
                 r,
                 ac_fn):
        super().__init__()
        self._ac_fn = ac_fn
        self._dim = int(in_c // r)
        self._shared_conv = ConvModule2D(in_c=in_c,
                                         out_c=self._dim,
                                         kernel_size=(1, 1),
                                         stride=(1, 1),
                                         dilation=(1, 1))
        self._shared_bn = nn.BatchNorm2d(self._dim)
        self._fconv = ConvModule2D(in_c=self._dim,
                                   out_c=in_c,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   dilation=(1, 1))
        self._tconv = ConvModule2D(in_c=self._dim,
                                   out_c=in_c,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   dilation=(1, 1))
        self._sigmoid = nn.Sigmoid()

    def forward(self, feature_maps):
        # batch_size, channel, frequency, time
        batch_size, channel, frequency, time = feature_maps.size()
        f = torch.mean(feature_maps, dim=-1)
        t = torch.mean(feature_maps, dim=-2)
        ft = torch.cat((f, t), dim=-1)
        ft_maps = self._shared_conv(ft.unsqueeze(-1))
        ft_maps = self._shared_bn(ft_maps)
        ft_maps = self._ac_fn(ft_maps)
        split_maps = torch.split(ft_maps,
                                 split_size_or_sections=[f.size()[-1], t.size()[-1]],
                                 dim=-2)
        f_maps = split_maps[0]
        t_maps = split_maps[1]
        t_maps = t_maps.view(batch_size, self._dim, 1, time)

        f_maps = self._fconv(f_maps)
        t_maps = self._tconv(t_maps)
        f_attention = self._sigmoid(f_maps)
        t_attention = self._sigmoid(t_maps)

        output = torch.multiply(feature_maps, t_attention)
        output = torch.multiply(output, f_attention)
        return output


class CTFALite(nn.Module):
    def __init__(self,
                 in_c,
                 ac_fn):
        super().__init__()
        self._ac_fn = ac_fn
        self._sigmoid = nn.Sigmoid()
        k = int(abs(math.log2(in_c)))
        k1 = k if k % 2 else k + 1
        self._conv = ConvModule2D(in_c=1,
                                  out_c=1,
                                  kernel_size=(k1, 1),
                                  stride=(1, 1),
                                  dilation=(1, 1))
        self._bn1 = nn.BatchNorm2d(in_c)
        self._bn2 = nn.BatchNorm2d(in_c)

    def forward(self, feature_maps):
        # batch_size, channel, frequency, time
        batch_size, channel, frequency, time = feature_maps.size()
        f_maps = torch.mean(feature_maps, dim=-1)  # b, c, f
        f_global = torch.mean(f_maps, dim=1, keepdim=True)  # b, 1, f
        t_maps = torch.mean(feature_maps, dim=-2)  # b, c, t
        t_global = torch.mean(t_maps, dim=1, keepdim=True)  # b, 1, t

        f_out = self._conv(f_maps.unsqueeze(dim=1))
        f = f_out.view(batch_size, channel, frequency, 1)
        f_sum = f + f_global.unsqueeze(dim=-1)
        f_weights = self._sigmoid(self._bn1(f_sum))

        t_out = self._conv(t_maps.unsqueeze(dim=1))
        t = t_out.view(batch_size, channel, 1, time)
        t_sum = t + t_global.unsqueeze(dim=-2)
        t_weights = self._sigmoid(self._bn2(t_sum))

        enhanced_out = torch.multiply(
            torch.multiply(feature_maps, f_weights), t_weights
        )

        return enhanced_out


class CTFALiteNoGlobal(nn.Module):
    def __init__(self,
                 in_c,
                 ac_fn):
        super().__init__()
        self._ac_fn = ac_fn
        self._sigmoid = nn.Sigmoid()
        k1 = int(abs(math.log2(in_c)))
        k2 = k1 if k1 % 2 else k1 + 1
        self._conv = ConvModule2D(in_c=1,
                                  out_c=1,
                                  kernel_size=(k2, 1),
                                  stride=(1, 1),
                                  dilation=(1, 1))
        self._bn1 = nn.BatchNorm2d(in_c)
        self._bn2 = nn.BatchNorm2d(in_c)

    def forward(self, feature_maps):
        # batch_size, channel, frequency, time
        batch_size, channel, frequency, time = feature_maps.size()
        f_maps = torch.mean(feature_maps, dim=-1)  # b, c, f
        t_maps = torch.mean(feature_maps, dim=-2)  # b, c, t

        f_out = self._conv(f_maps.unsqueeze(dim=1))
        f = f_out.view(batch_size, channel, frequency, 1)
        f_weights = self._sigmoid(self._bn1(f))

        t_out = self._conv(t_maps.unsqueeze(dim=1))
        t = t_out.view(batch_size, channel, 1, time)
        t_weights = self._sigmoid(self._bn2(t))

        enhanced_out = torch.multiply(
            torch.multiply(feature_maps, f_weights), t_weights
        )

        return enhanced_out


class CTFALiteNoT(nn.Module):
    def __init__(self,
                 in_c,
                 ac_fn):
        super().__init__()
        self._ac_fn = ac_fn
        self._sigmoid = nn.Sigmoid()
        k1 = int(abs(math.log2(in_c)))
        k2 = k1 if k1 % 2 else k1 + 1
        self._conv = ConvModule2D(in_c=1,
                                  out_c=1,
                                  kernel_size=(k2, 1),
                                  stride=(1, 1),
                                  dilation=(1, 1))
        self._bn1 = nn.BatchNorm2d(in_c)

    def forward(self, feature_maps):
        # batch_size, channel, frequency, time
        batch_size, channel, frequency, time = feature_maps.size()
        f_maps = torch.mean(feature_maps, dim=-1)  # b, c, f
        f_global = torch.mean(f_maps, dim=1, keepdim=True)  # b, 1, f

        f_out = self._conv(f_maps.unsqueeze(dim=1))
        f = f_out.view(batch_size, channel, frequency, 1)
        f_sum = f + f_global.unsqueeze(dim=-1)
        f_weights = self._sigmoid(self._bn1(f_sum))

        enhanced_out = torch.multiply(feature_maps, f_weights)

        return enhanced_out


class CTFALiteNoF(nn.Module):
    def __init__(self,
                 in_c,
                 ac_fn):
        super().__init__()
        self._ac_fn = ac_fn
        self._sigmoid = nn.Sigmoid()
        k1 = int(abs(math.log2(in_c)))
        k2 = k1 if k1 % 2 else k1 + 1
        self._conv = ConvModule2D(in_c=1,
                                  out_c=1,
                                  kernel_size=(k2, 1),
                                  stride=(1, 1),
                                  dilation=(1, 1))
        self._bn2 = nn.BatchNorm2d(in_c)

    def forward(self, feature_maps):
        # batch_size, channel, frequency, time
        batch_size, channel, frequency, time = feature_maps.size()
        t_maps = torch.mean(feature_maps, dim=-2)  # b, c, t
        t_global = torch.mean(t_maps, dim=1, keepdim=True)  # b, 1, t

        t_out = self._conv(t_maps.unsqueeze(dim=1))
        t = t_out.view(batch_size, channel, 1, time)
        t_sum = t + t_global.unsqueeze(dim=-2)
        t_weights = self._sigmoid(self._bn2(t_sum))

        enhanced_out = torch.multiply(feature_maps, t_weights)

        return enhanced_out

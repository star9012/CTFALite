import torch.nn as nn
from layers import ConvModule2D
from attention import CA, TFCBAM, SEModule, ECA, CTFALite
from attention import CTFALiteNoGlobal, CTFALiteNoT, CTFALiteNoF


class PreConvBlock(nn.Module):
    def __init__(self,
                 in_c,
                 num_filters,
                 kernel_size,
                 strides,
                 ac,
                 att_type,
                 r):
        super().__init__()
        self._ac = ac
        self._att_type = att_type
        filter1, filter2, filter3 = num_filters
        self._batch_norm1 = nn.BatchNorm2d(in_c)
        self._conv_layer1 = ConvModule2D(in_c=in_c,
                                         out_c=filter1,
                                         kernel_size=(1, 1),
                                         stride=strides,
                                         dilation=(1, 1))
        self._batch_norm2 = nn.BatchNorm2d(filter1)
        self._conv_layer2 = ConvModule2D(in_c=filter1,
                                         out_c=filter2,
                                         kernel_size=kernel_size,
                                         stride=(1, 1),
                                         dilation=(1, 1))
        self._batch_norm3 = nn.BatchNorm2d(filter2)
        self._conv_layer3 = ConvModule2D(in_c=filter2,
                                         out_c=filter3,
                                         kernel_size=kernel_size,
                                         stride=(1, 1),
                                         dilation=(1, 1))
        if att_type == r"CA":
            self._conv_att = CA(in_c=filter3,
                                r=r,
                                ac_fn=self._ac)
        elif att_type == r"TFCBAM":
            self._conv_att = TFCBAM(in_c=filter3,
                                    r=r,
                                    ac_fn=self._ac)
        elif att_type == r"CTFALite":
            self._conv_att = CTFALite(in_c=filter3,
                                      ac_fn=self._ac)
        elif att_type == r"SE":
            self._conv_att = SEModule(in_c=filter3,
                                      r=r,
                                      ac_fn=self._ac)
        elif att_type == r"ECA":
            self._conv_att = ECA(in_c=filter3)
        elif att_type == r"CTFALiteNoGlobal":
            self._conv_att = CTFALiteNoGlobal(in_c=filter3,
                                              ac_fn=self._ac)
        elif att_type == r"CTFALiteNoT":
            self._conv_att = CTFALiteNoT(in_c=filter3,
                                         ac_fn=self._ac)
        elif att_type == r"CTFALiteNoF":
            self._conv_att = CTFALiteNoF(in_c=filter3,
                                         ac_fn=self._ac)
        elif att_type == r"No":
            self._att_type = r"No"
        else:
            raise ValueError("Unknown att : {}".format(att_type))
        print(r"{} attention is used.".format(self._att_type))
        self._shortcut_conv_layer = ConvModule2D(in_c=in_c,
                                                 out_c=filter3,
                                                 kernel_size=(1, 1),
                                                 stride=strides,
                                                 dilation=(1, 1))

    def forward(self, input_data):
        x = self._batch_norm1(input_data)
        x = self._ac(x)
        x = self._conv_layer1(x)

        x = self._batch_norm2(x)
        x = self._ac(x)
        x = self._conv_layer2(x)

        x = self._batch_norm3(x)
        x = self._ac(x)
        x = self._conv_layer3(x)

        if self._att_type != r"No":
            x = self._conv_att(x)

        shortcut = self._shortcut_conv_layer(input_data)
        residual_output = x + shortcut
        return residual_output


class PreIdentityBlock(nn.Module):
    def __init__(self,
                 in_c,
                 num_filters,
                 kernel_size,
                 ac,
                 att_type,
                 r):
        super().__init__()
        self._ac = ac
        self._att_type = att_type
        filter1, filter2, filter3 = num_filters
        self._batch_norm1 = nn.BatchNorm2d(in_c)
        self._conv_layer1 = ConvModule2D(in_c=in_c,
                                         out_c=filter1,
                                         kernel_size=(1, 1),
                                         stride=(1, 1),
                                         dilation=(1, 1))
        self._batch_norm2 = nn.BatchNorm2d(filter1)
        self._conv_layer2 = ConvModule2D(in_c=filter1,
                                         out_c=filter2,
                                         kernel_size=kernel_size,
                                         stride=(1, 1),
                                         dilation=(1, 1))
        self._batch_norm3 = nn.BatchNorm2d(filter2)
        self._conv_layer3 = ConvModule2D(in_c=filter2,
                                         out_c=filter3,
                                         kernel_size=kernel_size,
                                         stride=(1, 1),
                                         dilation=(1, 1))
        if att_type == r"CA":
            self._conv_att = CA(in_c=filter3,
                                r=r,
                                ac_fn=self._ac)
        elif att_type == r"TFCBAM":
            self._conv_att = TFCBAM(in_c=filter3,
                                    r=r,
                                    ac_fn=self._ac)
        elif att_type == r"CTFALite":
            self._conv_att = CTFALite(in_c=filter3,
                                      ac_fn=self._ac)
        elif att_type == r"SE":
            self._conv_att = SEModule(in_c=filter3,
                                      r=r,
                                      ac_fn=self._ac)
        elif att_type == r"ECA":
            self._conv_att = ECA(in_c=filter3)
        elif att_type == r"CTFALiteNoGlobal":
            self._conv_att = CTFALiteNoGlobal(in_c=filter3,
                                              ac_fn=self._ac)
        elif att_type == r"CTFALiteNoT":
            self._conv_att = CTFALiteNoT(in_c=filter3,
                                         ac_fn=self._ac)
        elif att_type == r"CTFALiteNoF":
            self._conv_att = CTFALiteNoF(in_c=filter3,
                                         ac_fn=self._ac)
        elif att_type == r"No":
            self._att_type = r"No"
        else:
            raise ValueError("Unknown att : {}".format(att_type))
        print(r"{} attention is used.".format(self._att_type))

    def forward(self, input_data):
        x = self._batch_norm1(input_data)
        x = self._ac(x)
        x = self._conv_layer1(x)

        x = self._batch_norm2(x)
        x = self._ac(x)
        x = self._conv_layer2(x)

        x = self._batch_norm3(x)
        x = self._ac(x)
        x = self._conv_layer3(x)

        if self._att_type != r"No":
            x = self._conv_att(x)

        residual_output = x + input_data
        return residual_output


class PreResidualGroup(nn.Module):
    def __init__(self,
                 num_blocks,
                 in_c,
                 num_filters,
                 kernel_size,
                 strides,
                 ac,
                 att_type,
                 r):
        super().__init__()
        self._num_blocks = num_blocks
        self._ac = ac
        self._blocks = nn.ModuleList()
        filter1, filter2, filter3 = num_filters

        for i in range(self._num_blocks):
            if i == 0:
                conv_block = PreConvBlock(in_c=in_c,
                                          num_filters=num_filters,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          ac=self._ac,
                                          att_type=att_type,
                                          r=r)
                self._blocks.append(conv_block)
            else:
                self._blocks.append(
                    PreIdentityBlock(in_c=filter3,
                                     num_filters=num_filters,
                                     kernel_size=kernel_size,
                                     ac=self._ac,
                                     att_type=att_type,
                                     r=r)
                )

    def forward(self, input_data):
        x = input_data
        for j in range(len(self._blocks)):
            x = self._blocks[j](x)
        return x


class PreResnet50(nn.Module):
    def __init__(self,
                 ac,
                 att_type,
                 r):
        super().__init__()
        self._core_net = nn.Sequential(
            ConvModule2D(in_c=1,
                         out_c=32,
                         kernel_size=(7, 7),
                         stride=(2, 1),
                         dilation=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2)),
            PreResidualGroup(num_blocks=3,
                             in_c=32,
                             num_filters=[32, 32, 64],
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             ac=ac,
                             att_type=att_type,
                             r=r),
            PreResidualGroup(num_blocks=4,
                             in_c=64,
                             num_filters=[64, 64, 128],
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             ac=ac,
                             att_type=att_type,
                             r=r),
            PreResidualGroup(num_blocks=6,
                             in_c=128,
                             num_filters=[128, 128, 256],
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             ac=ac,
                             att_type=att_type,
                             r=r),
            PreResidualGroup(num_blocks=3,
                             in_c=256,
                             num_filters=[256, 256, 512],
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             ac=ac,
                             att_type=att_type,
                             r=r)
        )

    def forward(self, input_data):
        return self._core_net(input_data)

import torch.nn as nn
from layers import LinearModule
from front_end import PreResnet50
from loss import AAMSoftmax
from pooling import TP, TSP, SMHAP, MRMHAPooling


class FrameFeatureModule(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 ac):
        super().__init__()
        self._bn = nn.BatchNorm2d(in_c)
        self._ac = ac
        self._conv_layer = nn.Conv2d(in_channels=in_c,
                                     out_channels=out_c,
                                     kernel_size=(5, 1),
                                     stride=(1, 1),
                                     padding=(0, 0),
                                     padding_mode=r"zeros",
                                     dilation=(1, 1),
                                     groups=1,
                                     bias=False)

    def forward(self, input_data):
        x = self._conv_layer(self._ac(self._bn(input_data)))
        return x


class EmbeddingGenerator(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim):
        super().__init__()
        self._bn = nn.BatchNorm1d(in_dim)
        self._linear_layer = LinearModule(in_dim=in_dim,
                                          out_dim=out_dim)

    def forward(self, input_data):
        x = self._linear_layer(self._bn(input_data))
        return x


class MainModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self._frontend = PreResnet50(ac=nn.ReLU(),
                                     att_type=hparams.att_type,
                                     r=hparams.r)
        self._frame_generator = FrameFeatureModule(in_c=512,
                                                   out_c=512,
                                                   ac=nn.ReLU())
        if hparams.aggregation == r"temporal_pooling":
            self._aggregation_module = TP()
            self._embedding_generator = EmbeddingGenerator(in_dim=512,
                                                           out_dim=hparams.em_size)
        elif hparams.aggregation == r"statistical_pooling":
            self._aggregation_module = TSP()
            self._embedding_generator = EmbeddingGenerator(in_dim=1024,
                                                           out_dim=hparams.em_size)
        elif hparams.aggregation == r"mha_pooling":
            self._aggregation_module = SMHAP(in_dim=512,
                                             n_heads=8)
            self._embedding_generator = EmbeddingGenerator(in_dim=512,
                                                           out_dim=hparams.em_size)
        elif hparams.aggregation == r"mrmha_pooling":
            self._aggregation_module = MRMHAPooling(n_dim=512,
                                                    n_head=8)
            self._embedding_generator = EmbeddingGenerator(in_dim=512 * 8,
                                                           out_dim=hparams.em_size)            
        else:
            raise ValueError("Unknown aggregation method : {}"
                             .format(hparams.aggregation))

        self._criterion = AAMSoftmax(s=hparams.s,
                                     em_size=hparams.em_size,
                                     num_class=hparams.num_class)

    def forward(self, input_data, labels, updated_m, run_mode):
        frontend_output = self._frontend(input_data)
        frame_features = self._frame_generator(frontend_output)
        pooling_output = self._aggregation_module(frame_features)
        embedding = self._embedding_generator(pooling_output)
        if run_mode == r"test":
            return embedding
        elif run_mode == r"train":
            loss, acc = self._criterion(embedding, labels, updated_m, r"train")
            return loss, acc
        else:
            raise ValueError("Unknown mode : {}.".format(run_mode))

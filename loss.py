import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()
        self._softmax_loss = nn.CrossEntropyLoss()

    def forward(self,
                predictions: torch.FloatTensor,
                targets: torch.LongTensor):
        targets.requires_grad = False
        loss = self._softmax_loss(predictions, targets)
        correct = torch.eq(targets, predictions.argmax(dim=-1)).float().sum()
        total_num = predictions.size()[0]
        acc = correct / total_num
        return loss, acc


class AAMSoftmax(nn.Module):
    def __init__(self, em_size, num_class, s):
        super(AAMSoftmax, self).__init__()
        self._ce = nn.CrossEntropyLoss()
        self._em_size = em_size
        self._num_class = num_class
        self._s = s
        self._class_weights = nn.Parameter(torch.FloatTensor(em_size, num_class))
        nn.init.xavier_uniform_(self._class_weights)

    def forward(self,
                input_data,
                labels,
                margin,
                mode):
        assert mode in [r"train", r"vis"], print("Unknown mode for loss computing: {}".format(mode))
        labels.requires_grad = False
        normalized_input = F.normalize(input_data, p=2, dim=1)
        normalized_weights = F.normalize(self._class_weights, p=2, dim=0)
        cosine = torch.mm(normalized_input, normalized_weights)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        cos_m = math.cos(margin)
        sin_m = math.sin(margin)
        threshold = math.cos(math.pi - margin)
        cos_with_m = cosine * cos_m - sine * sin_m
        cos_with_m = torch.where(cosine > threshold,
                                 cos_with_m,
                                 cosine - margin * math.sin(math.pi - margin))
        one_hots = torch.zeros(cosine.size())
        if torch.cuda.is_available():
            one_hots = one_hots.cuda()
        one_hots.scatter_(1, labels.view(-1, 1).long(), 1)
        if mode == r"train":
            logits = (one_hots * cos_with_m) + ((1.0 - one_hots) * cosine)
            s_logits = self._s * logits
            aam_loss = self._ce(s_logits, labels)
            correct = torch.eq(labels, logits.argmax(dim=-1)).float().sum()
            total_num = logits.size(0)
            acc = correct / total_num
            return aam_loss, acc
        else:
            raise ValueError("mode for loss computing is unknown.")

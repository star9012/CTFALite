import os
import torch
from model import MainModel


def to_gpu(x):
    x = x.contiguous()
    x = x.cuda(non_blocking=True)
    x.requires_grad = False
    return x


def count_parameters(model):
    total_num = sum([p.numel() for p in model.parameters()])
    trainable_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Number of parameters, Total : {}, Trainable : {}".format(total_num, trainable_num))


def initialize_model(hparams):
    model = MainModel(hparams=hparams)
    if torch.cuda.is_available():
        model.cuda()
    return model


def save_checkpoint(model, optimizer, iteration, save_path):
    print("At iteration {}, saving model and optimizer to checkpoint {}."
          .format(iteration, save_path))
    torch.save({'iteration': iteration,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               save_path)


def validate_spk(file_path, data_dir):
    with open(file_path, mode=r"r") as f:
        file_content = f.readlines()
    ids = [line.strip() for line in file_content]
    ids = set(ids)
    print(ids - set(os.listdir(data_dir)))


if __name__ == r"__main__":
    validate_spk(r"/home/dell/Yuheng/FrameAggregation/spk_id.txt",
                 r"/mnt/32da6dad-b2d9-45a9-8959-49fff09a3aa3/weiyh/vox2")

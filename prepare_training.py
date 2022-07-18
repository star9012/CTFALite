import os
import logging
import numpy as np
from glob import glob
from logger import LogRecoder
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import TrainingDataset, TrainingDataCollate


def create_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
        print("------ Created Directory : {}".format(directory))


def create_training_dirs(save_root_dir):
    create_dir(os.path.join(save_root_dir, r"dict"))
    create_dir(os.path.join(save_root_dir, r"spk"))
    create_dir(os.path.join(save_root_dir, r"log"))
    create_dir(os.path.join(save_root_dir, r"tensorboard"))
    create_dir(os.path.join(save_root_dir, r"checkpoint"))


def prepare_training_data(data_root_dir,
                          spk_datafiles,
                          spk_id_dict,
                          hparams):
    train_set = TrainingDataset(data_root_dir,
                                spk_datafiles,
                                hparams,
                                spk_id_dict)
    collate_fn = TrainingDataCollate(hparams)

    train_loader = DataLoader(train_set,
                              num_workers=hparams.num_works,
                              shuffle=True,
                              batch_size=hparams.batch_size,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)
    return train_loader


def prepare_spkfiles(data_root_dir,
                     spk_save_dir,
                     dict_save_dir):
    id_map_path = os.path.join(dict_save_dir, r"spk_dict.npy")
    if not os.path.isdir(spk_save_dir) or not os.path.isfile(id_map_path):
        spk_dict = dict()
        # data_root_dir/spk_dir/wav_files
        wave_files = glob(os.path.join(data_root_dir, r"*", r"*.wav"))
        for wave_file in wave_files:
            spk_name = wave_file.split(os.sep)[-2]
            if spk_name not in spk_dict:
                spk_dict.update({spk_name: []})
            wave_file_name = wave_file.split(os.sep)[-1]
            spk_dict[spk_name].append(os.path.join(spk_name, wave_file_name))

        name_to_id = dict()
        for i, key in enumerate(sorted(spk_dict.keys())):
            save_path = os.path.join(spk_save_dir, "{}.npy".format(key))
            np.save(save_path, np.array(spk_dict[key]))
            print("{}".format(save_path))
            name_to_id.update({key: i})
        assert len(name_to_id) == len(spk_dict.keys())
        print("------ There are {} training speakers.".format(len(name_to_id)))
        np.save(id_map_path, name_to_id)


def prepare_log(log_directory):
    train_logger = LogRecoder(logger_name="TrainLogger",
                              log_file_path=os.path.join(log_directory, r"train_log.log"),
                              std_handler_level=logging.WARNING,
                              file_handler_level=logging.INFO)
    return train_logger


def prepare_tensorboardx(tensorboard_dir):
    writer = SummaryWriter(tensorboard_dir)
    return writer

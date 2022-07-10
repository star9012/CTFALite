import os
import torch
import numpy as np
from utils import to_gpu
from dataset import load_wav
from evaluate import load_model
from evaluate import calculate_eer
from dataset import extract_melspectrogram
from torch.utils.data import DataLoader, Dataset


class CnCelebTest(Dataset):
    def __init__(self,
                 trial_path):
        super(CnCelebTest, self).__init__()
        self._wave_files = []
        with open(trial_path, "r") as f:
            for trial in f.readlines():
                anchor, test = trial.strip().split(r" ")[-2:]
                self._wave_files.append(anchor)
                self._wave_files.append(test)
        self._wave_file_unique = np.unique(np.array(self._wave_files))
        print(self._wave_file_unique)

    def __getitem__(self, index):
        wave_file = self._wave_file_unique[index]
        data = load_wav(audio_file=wave_file)
        label = os.sep.join(wave_file.split(os.sep)[-2:])
        return data, label

    def __len__(self):
        return len(self._wave_file_unique)


class TestCollate:
    def __init__(self, hp):
        self.hp = hp

    def __call__(self, batch):
        data, label = batch[0]
        features = extract_melspectrogram(data, self.hp)
        features = features.reshape(shape=(1, 1, features.size()[-2], features.size()[-1]))
        features.requires_grad = False
        return features, label


def load_data(trial_path, hparams):
    test_set = CnCelebTest(trial_path=trial_path)
    test_collate = TestCollate(hp=hparams)
    data_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=1,
                             drop_last=False,
                             pin_memory=False,
                             collate_fn=test_collate,
                             num_workers=8)
    return data_loader


def extract_embedding(hp,
                      checkpoint_path,
                      trial_path):
    model, iteration = load_model(hparams=hp, restore_path=checkpoint_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    ems = {}
    with torch.no_grad():
        test_data_loader = load_data(trial_path=trial_path,
                                     hparams=hp)
        for data, label in test_data_loader:
            if torch.cuda.is_available():
                data = to_gpu(data)
            em = model(data, None, None, r"test")
            if label not in ems.keys():
                ems.update({label: em.cpu().detach().numpy()})
                print("Generated embedding for {}.".format(label))

    return ems, iteration


def evaluate_performance(hp,
                         trial_file,
                         checkpoint_path):

    ems, iteration = extract_embedding(hp=hp,
                                       checkpoint_path=checkpoint_path,
                                       trial_path=trial_file)

    enroll_ems, test_ems, ground_truth = [], [], []
    with open(trial_file, mode="r") as f:
        for line in f:
            elements = line.strip().split(r" ")
            enroll_em = ems[os.sep.join(elements[1].split(os.sep)[-2:])]
            enroll_ems.append(enroll_em)
            test_em = ems[os.sep.join(elements[2].split(os.sep)[-2:])]
            test_ems.append(test_em)
            ground_truth.append(int(elements[0]))
    anchor_embeddings = np.concatenate(enroll_ems, axis=0)
    test_embeddings = np.concatenate(test_ems, axis=0)
    eer = calculate_eer(anchor_embeddings=anchor_embeddings,
                        test_embeddings=test_embeddings,
                        ground_truth=ground_truth)
    return eer, iteration


if __name__ == r"__main__":
    from hyper_parameters import HyperParams
    hparams = HyperParams()
    SAVE_ROOT_DIR = hparams.save_root_dir
    TEST_ROOT_DIR = hparams.test_root_dir
    CHECKPOINT_DIR = os.path.join(SAVE_ROOT_DIR, r"checkpoint")
    TRIAL_FILE = hparams.cn_trial_file
    AGGREGATION = hparams.aggregation
    # Load checkpoint if one exists
    restore_path = os.path.join(CHECKPOINT_DIR, r"checkpoint-{}".format(AGGREGATION))
    eer, eval_iteration = evaluate_performance(trial_file=TRIAL_FILE,
                                               checkpoint_path=restore_path,
                                               hp=hparams)
    print(CHECKPOINT_DIR)
    print(r"EER: {}".format(eer))

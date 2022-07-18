import os
import torch
import numpy as np
from utils import to_gpu
from torch.utils.data import DataLoader
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize
from dataset import TestDataset, TestCollateData
from utils import initialize_model


def load_model(hparams, restore_path):
    model = initialize_model(hparams)
    if os.path.isfile(restore_path):
        checkpoint_dict = torch.load(restore_path, map_location='cpu')
        model.load_state_dict(checkpoint_dict['model'])
        iteration = checkpoint_dict['iteration']
        print("Loaded checkpoint '{}' from iteration {}.".format(restore_path, iteration))
        return model, iteration
    else:
        raise FileNotFoundError("Not found file : {}".format(restore_path))


def load_dataloader(anchor_files, test_files, hparams):
    test_set = TestDataset(anchor_files=anchor_files,
                           test_files=test_files,
                           hparams=hparams)
    cfn = TestCollateData(hparams)
    data_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=1,
                             drop_last=False,
                             pin_memory=False,
                             collate_fn=cfn,
                             num_workers=hparams.num_works)
    return data_loader


def calculate_cosine(anchor_embeddings, test_embeddings):
    normalized_anchor = normalize(anchor_embeddings, norm='l2', axis=1)
    normalized_test = normalize(test_embeddings, norm='l2', axis=1)
    multiply_result = np.multiply(normalized_anchor, normalized_test)
    dot_result = np.sum(multiply_result, axis=-1, keepdims=False)
    return dot_result


def calculate_eer(anchor_embeddings, test_embeddings, ground_truth):
    cosine_similarity = calculate_cosine(anchor_embeddings, test_embeddings)
    fpr, tpr, thresholds = roc_curve(ground_truth, cosine_similarity, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    print("EER: {}".format(eer))
    print("EER Threshold: {}".format(thresh))
    return eer


def generate_embedding(checkpoint_file,
                       anchor_files,
                       test_files,
                       hparams):

    all_anchor_embeddings = list()
    all_test_embeddings = list()

    model, iteration = load_model(hparams=hparams, restore_path=checkpoint_file)
    model.eval()

    with torch.no_grad():
        test_data_loader = load_dataloader(anchor_files=anchor_files,
                                           test_files=test_files,
                                           hparams=hparams)
        for i, data_tuple in enumerate(test_data_loader):
            anchor_mel, test_mel = data_tuple
            if torch.cuda.is_available():
                anchor_mel = to_gpu(anchor_mel)
                test_mel = to_gpu(test_mel)
            anchor_embedding = model(anchor_mel, None, None, r"test")
            test_embedding = model(test_mel, None, None, r"test")
            all_anchor_embeddings.append(anchor_embedding.cpu().detach().numpy())
            all_test_embeddings.append(test_embedding.cpu().detach().numpy())
            print("-------- Generated Embeddings for {}-th Batch. --------".format(i))
        all_anchor_embeddings = np.concatenate(all_anchor_embeddings, axis=0)
        all_test_embeddings = np.concatenate(all_test_embeddings, axis=0)
        return all_anchor_embeddings, all_test_embeddings, iteration


def evaluate_performance(testdata_root_dir,
                         trial_file,
                         checkpoint_path,
                         hparams):
    print("---------------- Start to generate embeddings for positive trials ----------------")
    anchor_files, test_files, ground_truth = [], [], []
    with open(trial_file) as f:
        for line in f:
            elements = line.strip().split(r" ")
            anchor_file = os.path.join(elements[1].split(r"/")[0],
                                       elements[1].split(r"/")[1], elements[1].split(r"/")[2])
            anchor_files.append(os.path.join(testdata_root_dir, anchor_file))
            test_file = os.path.join(elements[2].split(r"/")[0],
                                     elements[2].split(r"/")[1], elements[2].split(r"/")[2])
            test_files.append(os.path.join(testdata_root_dir, test_file))
            ground_truth.append(int(elements[0]))

    a_ems, t_ems, iteration = generate_embedding(checkpoint_file=checkpoint_path,
                                                 anchor_files=anchor_files,
                                                 test_files=test_files,
                                                 hparams=hparams)
    eer = calculate_eer(anchor_embeddings=a_ems,
                        test_embeddings=t_ems,
                        ground_truth=ground_truth)
    return eer, iteration


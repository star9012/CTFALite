import os
import torch
import numpy as np
from glob import glob
from evaluate import evaluate_performance
from utils import to_gpu, initialize_model, save_checkpoint
from prepare_training import prepare_training_data, prepare_spkfiles
from prepare_training import create_training_dirs, prepare_log, prepare_tensorboardx


def train(hparams):

    TRAIN_ROOT_DIR = hparams.train_root_dir
    TEST_ROOT_DIR = hparams.test_root_dir
    SAVE_ROOT_DIR = hparams.save_root_dir
    DICT_DIR = os.path.join(SAVE_ROOT_DIR, r"dict")
    SPK_DIR = os.path.join(SAVE_ROOT_DIR, r"spk")
    LOG_DIR = os.path.join(SAVE_ROOT_DIR, r"log")
    BOARD_DIR = os.path.join(SAVE_ROOT_DIR, r"tensorboard")
    CHECKPOINT_DIR = os.path.join(SAVE_ROOT_DIR, r"checkpoint")

    MAX_M = hparams.m

    INITIAL_LR = hparams.initial_lr
    MAX_EPOCHS = hparams.max_epoch
    DECAY_EPOCHS = hparams.decay_epochs

    TEST_MODE = hparams.test_during_training
    TRIAL_FILE = hparams.trial_file
    SAVE_INTERVAL = hparams.save_interval

    AGGREGATION = hparams.aggregation

    # create dirs, logger and tensorboard writer
    create_training_dirs(SAVE_ROOT_DIR)
    logger = prepare_log(LOG_DIR)
    tx_writer = prepare_tensorboardx(BOARD_DIR)

    # model and optimizer
    model = initialize_model(hparams)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=INITIAL_LR,
                                 weight_decay=1e-4)

    iteration = 0
    # Load checkpoint if one exists
    restore_path = os.path.join(CHECKPOINT_DIR, r"checkpoint-{}".format(AGGREGATION))
    if os.path.isfile(restore_path):
        checkpoint_dict = torch.load(restore_path, map_location='cpu')
        model.load_state_dict(checkpoint_dict['model'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        iteration = checkpoint_dict['iteration']
        print("Loaded checkpoint '{}' from iteration {}.".format(restore_path, iteration))
        iteration += 1  # next iteration is iteration + 1

    # create data loader
    prepare_spkfiles(data_root_dir=TRAIN_ROOT_DIR,
                     spk_save_dir=SPK_DIR,
                     dict_save_dir=DICT_DIR)
    spkid_dict_path = os.path.join(DICT_DIR, r"spk_dict.npy")
    train_loader = prepare_training_data(data_root_dir=TRAIN_ROOT_DIR,
                                         spk_datafiles=glob(os.path.join(SPK_DIR, r"id*.npy")),
                                         spk_id_dict=np.load(spkid_dict_path, allow_pickle=True).item(),
                                         hparams=hparams)

    model.train()

    epoch = int(iteration // len(train_loader))
    step = int(iteration % len(train_loader))
    while epoch < MAX_EPOCHS:
        optimizer.param_groups[0]['lr'] = INITIAL_LR * 0.1 ** (epoch // DECAY_EPOCHS)
        epoch_rate = epoch / MAX_EPOCHS
        updated_factor = 2. / (1. + np.exp(-10.0 * epoch_rate)) - 1
        data_iterator = iter(train_loader)
        while step < len(train_loader):
            optimizer.zero_grad()
            features, spk_labels = next(data_iterator)
            if torch.cuda.is_available():
                features = to_gpu(features)
                spk_labels = to_gpu(spk_labels)
            spk_loss, acc = model(features, spk_labels, updated_factor * MAX_M, r"train")
            spk_loss.backward()
            optimizer.step()
            step += 1

            logger.warning("Epoch : {}, Iteration : {}, LR : {}, Margin : {}"
                           .format(epoch, iteration, optimizer.param_groups[0]['lr'], updated_factor * MAX_M))
            logger.warning("Loss : {}, Acc : {}".format(spk_loss.item(), acc))
            tx_writer.add_scalar("{}-Acc".format(AGGREGATION), acc, global_step=iteration)
            tx_writer.add_scalar("{}-loss".format(AGGREGATION), spk_loss.item(), global_step=iteration)

            if iteration % SAVE_INTERVAL == 0:
                save_checkpoint(model=model,
                                optimizer=optimizer,
                                iteration=iteration,
                                save_path=restore_path)
            iteration += 1

        if TEST_MODE and epoch % 20 == 0 and epoch > 0:
            eer, eval_iteration = evaluate_performance(trial_file=TRIAL_FILE,
                                                       checkpoint_path=restore_path,
                                                       hparams=hparams,
                                                       testdata_root_dir=TEST_ROOT_DIR)
            if eer is not None:
                tx_writer.add_scalar("{}-EER".format(AGGREGATION),
                                     eer,
                                     global_step=eval_iteration)
        epoch += 1
        step = 0


if __name__ == r"__main__":
    from hyper_parameters import HyperParams
    hyper_params = HyperParams()
    train(hyper_params)

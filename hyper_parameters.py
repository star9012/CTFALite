import os


class HyperParams:
    def __init__(self):
        # acoustic feature
        self.sampling_rate = 16000
        self.hop_length = 160
        self.win_length = 320
        self.n_fft = 320

        # task
        self.em_size = 256
        self.num_class = 1211

        # mini-batch
        self.batch_size = 64
        self.max_duration = 2.5

        # logger
        self.log_steps = 10

        # train
        self.max_epoch = 160
        self.initial_lr = 1e-3
        self.decay_epochs = 40
        self.save_interval = 10000
        self.num_works = 4
        self.test_during_training = True

        # loss
        self.s = 40.0
        self.m = 0.15

        # convolutional attention and aggregation
        # six different convolutional attention models: No, SE, ECA, TFCBAM, CA, and, CTFALite
        self.att_type = r"CA"
        self.r = 4
        # four different pooling layers: temporal_pooling, statistical_pooling, mha_pooling, mrmha_pooling
        self.aggregation = r"mrmha_pooling"

        # dir
        self.train_root_dir = r"\YourPath\VoxCeleb1_Train"
        self.test_root_dir = r"\YourPath\VoxCeleb1_Test"
        self.save_root_dir = os.path.join(r"your directory path used for saving training files",
                                          r"_".join((self.aggregation, self.att_type, str(self.r))))
        self.trial_file = r"veri_test.txt"

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
        self.num_works = 8
        self.test_during_training = True

        # loss
        self.s = 40.0
        self.m = 0.15

        # frond-end attention and aggregation
        self.att_type = r"CA"
        self.r = 32
        self.aggregation = r"mrmha_pooling"

        # dir
        self.train_root_dir = r"/home/lthpc/weiyh/Vox1/train"
        self.test_root_dir = r"/media/lthpc/a2467eb3-89a3-4919-9f8a-f89428e9698e/weiyh/vox1_test"
        self.save_root_dir = os.path.join(r"/media/lthpc/affcc524-0a02-47b5-83e5-faa9c9ef0585/backup/yuheng/icassp2022/checkpoint1",
                                          r"_".join((self.aggregation, self.att_type, str(self.r), str(self.m), str(self.decay_epochs))))
        self.trial_file = r"/media/lthpc/affcc524-0a02-47b5-83e5-faa9c9ef0585/backup/yuheng/veri_test.txt"
        self.cn_trial_file = r"/home/lthpc/weiyh/CNCeleb/trials.txt"


import os
from evaluate import evaluate_performance


def test(hparams):
    SAVE_ROOT_DIR = hparams.save_root_dir
    TEST_ROOT_DIR = hparams.test_root_dir
    CHECKPOINT_DIR = os.path.join(SAVE_ROOT_DIR, r"checkpoint")
    TRIAL_FILE = hparams.trial_file
    AGGREGATION = hparams.aggregation
    # Load checkpoint if one exists
    restore_path = os.path.join(CHECKPOINT_DIR, r"checkpoint-{}".format(AGGREGATION))
    eer, eval_iteration = evaluate_performance(trial_file=TRIAL_FILE,
                                               checkpoint_path=restore_path,
                                               hparams=hparams,
                                               testdata_root_dir=TEST_ROOT_DIR)
    print(eer)


if __name__ == r"__main__":
    from hyper_parameters import HyperParams
    hyper_params = HyperParams()
    test(hyper_params)

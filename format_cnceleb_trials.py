import os
import argparse
import numpy as np


def format_cnceleb_trials(cnceleb1_root,
                          trial_save_path):
    enroll_lst_path = os.path.join(cnceleb1_root, "eval", "lists", "enroll.lst")
    raw_trl_path = os.path.join(cnceleb1_root, "eval", "lists", "trials.lst")

    spk2wav_mapping = {}
    enroll_lst = np.loadtxt(enroll_lst_path, str)
    for item in enroll_lst:
        spk2wav_mapping[item[0]] = item[1]
    trials = np.loadtxt(raw_trl_path, str)

    with open(trial_save_path, "w") as f:
        for item in trials:
            enroll_path = os.path.join(cnceleb1_root, "eval", spk2wav_mapping[item[0]])
            test_path = os.path.join(cnceleb1_root, "eval", item[1])
            label = item[2]
            f.write("{} {} {}\n".format(label, enroll_path, test_path))


if __name__ == r"__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(r"--cnceleb1_root", type=str, required=True)
    args = parser.parse_args()
    format_cnceleb_trials(args.cnceleb1_root, r"cnceleb_trials.txt")

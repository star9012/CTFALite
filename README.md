# CTFALite: Lightweight Channel-specific Temporal and Frequency Attention Mechanism for Enhancing the Speaker Embedding Extractor (interspeech2022)
This is a Pytorch-based implementation of our work [CTFALite](https://www.isca-speech.org/archive/interspeech_2022/wei22d_interspeech.html). All convolutional attention models are implemented in attention.py.

The convolution kernel size $k$ used in CTFALite can be set to any value as you want, such as $k=\lfloor0.5\log_{2}N_{c}+0.5\rfloor$ (the same as ECA) or a fixed value (3, 5, etc.).

Cite as: Wei, Y., Du, J., Liu, H., Wang, Q. (2022) CTFALite: Lightweight Channel-specific Temporal and Frequency Attention Mechanism for Enhancing the Speaker Embedding Extractor. Proc. Interspeech 2022, 341-345, doi: 10.21437/Interspeech.2022-10288

@inproceedings{wei22d_interspeech,
  author={Yuheng Wei and Junzhao Du and Hui Liu and Qian Wang},
  title={{CTFALite: Lightweight Channel-specific Temporal and Frequency Attention Mechanism for Enhancing the Speaker Embedding Extractor}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={341--345},
  doi={10.21437/Interspeech.2022-10288}
}

## Datasets
The datasets used in our experiments include the training and test sets of VoxCeleb1 and the test set of CN-Celeb1, which can be downloaded from [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb) and [Cn-Celeb1](https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz). 
## Preparation
1. python==3.7 pytorch==1.7.0 torchaudio==0.7.0 tensorboardx==2.2 scikit-learn==1.0.2 scipy==1.6.2
2. make sure that all audio files are in .wav format.  
3. make sure that the directory structures of the training and test sets of VoxCeleb1 are organized as *\YourPath\VoxCeleb1_Train\Speakers\Wavfiles* and *\YourPath\VoxCeleb1_Test\Speakers\Wavefiles*, respectively.
## Running
### Train models
1. set the values of hyperparameters in hyper_parameters.py.
2. Train models with the training set of VoxCeleb1 and evaluate them with the official trials provided by VoxCeleb1: `python train.py`
3. EER is recorded in a tensorboard file. You can find it in the *tensorboard* subdirectory of the hyperparameter *save_root_dir*, and the trained model is saved in the *checkpoint* subdirectory.
### Test models with the trials of CN-Celeb1
1. Suppose your directory structure of CN-Celeb1 is *\YourPath\CN-Celeb1\eval\···*. Run `python format_cnceleb_trials.py --cnceleb1_root \YourPath\CN-Celeb1` to formalize the trial file. This generates a file named as *cnceleb_trials.txt* in the current directory.
2. Run `python cnceleb.py --restore_path [your path of a checkpoint file w.r.t a model]` to evaluate a trained model.

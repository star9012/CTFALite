# CTFALite: Lightweight Channel-specific Temporal and Frequency Attention Mechanism for Enhancing the Speaker Embedding Extractor (interspeech2022)
This is a Pytorch-based implementation of our work [CTFALite](). Packages are listed in requirements.txt.
## Datasets
The datasets used in our experiments include the training and test sets of VoxCeleb1 and the test set of CN-Celeb1, which can be downloaded from [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb) and [Cn-Celeb1](https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz). 
## Preparation
1. make sure that all audio files are in .wav format.  
2. make sure that the directory structures of the training and test sets of VoxCeleb1 are organized as *\YourPath\VoxCeleb1_Train\Speakers\Wavfiles* and *\YourPath\VoxCeleb1_Test\Speakers\Wavefiles*, respectively.
## Running
### Train models
1. set the values of hyperparameters in hyper_parameters.py.
2. Train models with the training set of VoxCeleb1 and evaluate them with the official trials provided by VoxCeleb1: `python train.py`
3. EER is recorded in a tensorboard file. You can find it in the *tensorboard* subdirectory of the hyperparameter *save_root_dir*, and the trained model is saved in the *checkpoint* subdirectory.
### Test models with the trials of CN-Celeb1
1. Suppose your directory structure of CN-Celeb1 is *\YourPath\CN-Celeb1\eval\···*. Run `python format_cnceleb_trials.py --cnceleb1_root \YourPath\CN-Celeb1` to formalize the trial file. This generates a file named as *cnceleb_trials.txt* in the current directory.
2. Run `python cnceleb.py --restore_path [your path of a checkpoint file w.r.t a model]` to evaluate a trained model.

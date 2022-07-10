# CTFALite: Lightweight Channel-specific Temporal and Frequency Attention Mechanism for Enhancing the Speaker Embedding Extractor (interspeech2022)
This is a Pytorch-based implementation of our work [CTFALite](). Packages are listed in requirements.txt.
## Datasets
The datasets used in our experiments include the training and test sets of VoxCeleb1 and the test set of CN-Celeb1, which can be downloaded from [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb) and [Cn-Celeb1](https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz). 
## Preparation
1. make sure that all audio files are in .wav format.  
2. make sure that the directory structures of the training and test sets of VoxCeleb1 are organized as \YourPath\VoxCeleb1_Train\Speakers\Wavfiles and \YourPath\VoxCeleb1_Test\Speakers\Wavefiles, respectively.
## Running
1. Train models with the training set of VoxCeleb1 and evaluate them with the official trials provided by VoxCeleb1:
·python train.py·

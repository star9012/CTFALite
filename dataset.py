import os
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset


def load_wav(audio_file):
    sample_rate, data = wavfile.read(audio_file)
    return data


def extract_speech_segment(speech_signal, max_len):
    if len(speech_signal) > max_len:
        start_ = np.random.randint(low=0, high=len(speech_signal) - max_len + 1, size=1)[0]
        speech_seg = speech_signal[start_:start_ + max_len]
    else:
        speech_seg = speech_signal
    return speech_seg


def extract_melspectrogram(signal, hparams):
    eps = np.finfo(float).eps
    signal_tensor = torch.FloatTensor(signal)
    signal_tensor.requires_grad = False
    """
    fn = torchaudio.transforms.MelSpectrogram(sample_rate=hparams.sampling_rate,
                                              n_fft=hparams.n_fft,
                                              win_length=hparams.win_length,
                                              hop_length=hparams.hop_length,
                                              f_min=hparams.mel_fmin,
                                              f_max=hparams.mel_fmax,
                                              n_mels=hparams.n_mel_channels,
                                              power=2.0)
    """
    fn = torchaudio.transforms.Spectrogram(n_fft=hparams.n_fft,
                                           win_length=hparams.win_length,
                                           hop_length=hparams.hop_length,
                                           power=1)
    spectrogram = fn(signal_tensor)
    mean = torch.mean(spectrogram, dim=-1, keepdim=True)
    std = torch.std(spectrogram, dim=-1, keepdim=True)
    normalized_spectrogram = (spectrogram - mean) / (std + eps)
    return normalized_spectrogram


def extract_feature_from_signal(wav_file_path, hparams):
    max_len = int(hparams.max_duration * hparams.sampling_rate)
    waveform = load_wav(wav_file_path)
    wave_seg = extract_speech_segment(waveform, max_len)
    acoustic_feature = extract_melspectrogram(wave_seg, hparams)
    return acoustic_feature


class TrainingDataset(Dataset):
    def __init__(self,
                 data_root_dir,
                 spk_datafiles,
                 hparams,
                 spk_id_dict):
        self._data_root_dir = data_root_dir
        self._datafiles = []
        for spk_datafile in spk_datafiles:
            self._datafiles += list(np.load(spk_datafile))
        self._spk_id_dict = spk_id_dict
        self._hparams = hparams

    def __getitem__(self, index):
        wave_file = self._datafiles[index]
        wav_file_path = os.path.join(self._data_root_dir, wave_file)
        acoustic_feature = extract_feature_from_signal(wav_file_path, self._hparams)
        speaker_name = wave_file.split(os.sep)[-2]
        speaker_label = self._spk_id_dict[speaker_name]
        return [acoustic_feature, speaker_label]

    def __len__(self):
        return len(self._datafiles)


class TrainingDataCollate:
    def __init__(self, hparams):
        self._hparams = hparams

    def __call__(self, data):
        cum_features = []
        cum_labels = []
        for data_tuple in data:
            cum_features.append(data_tuple[0])
            cum_labels.append(data_tuple[1])
        batch_size = len(cum_features)
        feature_tensor = torch.FloatTensor(batch_size,
                                           1,
                                           cum_features[0].size()[0],
                                           cum_features[0].size()[1])
        feature_tensor.zero_()
        label_tensor = torch.LongTensor(batch_size)
        label_tensor.zero_()
        for i in range(batch_size):
            feature_tensor[i, 0, :, :] = cum_features[i]
            label_tensor[i] = cum_labels[i]
        feature_tensor.requires_grad = False
        label_tensor.requires_grad = False
        return [feature_tensor, label_tensor]


class TestDataset(Dataset):
    def __init__(self,
                 anchor_files,
                 test_files,
                 hparams):
        super(TestDataset, self).__init__()
        self._hparams = hparams
        self._anchor_data = anchor_files
        self._test_data = test_files

    def __getitem__(self, index):
        print("index")
        print(index)
        anchor_file = self._anchor_data[index]
        test_file = self._test_data[index]
        print("anchor file")
        print(anchor_file)
        print("test file")
        print(test_file)
        print("-------------------------------------")
        anchor_signal = load_wav(anchor_file)
        test_signal = load_wav(test_file)
        return [anchor_signal, test_signal]

    def __len__(self):
        return len(self._test_data)


class TestCollateData:
    def __init__(self, hparams):
        self._hparams = hparams
    
    def __call__(self, data):
        anchor_features = extract_melspectrogram(data[0][0], self._hparams)
        test_features = extract_melspectrogram(data[0][1], self._hparams)
        anchor_features = anchor_features.reshape(shape=(1, 1, anchor_features.size()[-2], anchor_features.size()[-1]))
        anchor_features.requires_grad_(False)
        test_features = test_features.reshape(shape=(1, 1, test_features.size()[-2], test_features.size()[-1]))
        test_features.requires_grad_(False)
        return [anchor_features, test_features]

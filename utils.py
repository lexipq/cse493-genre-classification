import argparse
import librosa
import librosa.display
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


spec_dim = 128
genre_to_label = {
    'blues': 0,
    'classical': 1,
    'country': 2,
    'disco': 3,
    'hiphop': 4,
    'jazz': 5,
    'metal': 6,
    'pop': 7,
    'reggae': 8,
    'rock': 9
}


def load_mlp_data(train_split=0.8):
    x_path = 'data/mlp_data.pt'
    y_path = 'data/mlp_gt.pt'
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        df = pd.read_csv('data/features_3_sec.csv')
        df = df.drop(labels='filename', axis=1)

        # get all target classes and convert -> index [0-9]
        classes = df.iloc[:,-1]
        converter = LabelEncoder()
        y = converter.fit_transform(classes)
        y = torch.tensor(y, dtype=torch.long)

        # standardize all features: zero mean unit variance
        norm = StandardScaler()
        x = norm.fit_transform(np.array(df.iloc[:,:-1], dtype=np.float32))
        x = torch.from_numpy(x)
        torch.save(x, x_path)
        torch.save(y, y_path)
    else:
        x = torch.load(x_path)
        y = torch.load(y_path)

    print('loaded tensor x:', x.shape)
    print('loaded tensor y', y.shape)
    return train_test_split(x, y, train_size=train_split)


def load_sampled_cnn_data(sample_size=3, train_split=0.8):
    x_path = 'data/cnn_data.pt'
    y_path = 'data/cnn_gt.pt'
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        dataset = []
        labels = []
        base = 'data/genres/'

        for root, _, files in os.walk(base):
            if root == base:
                continue

            genre = root[len(base):]
            print('genre loaded:', genre)
            label = genre_to_label[genre]

            for file in files:
                # skip corrupted file
                if file == 'jazz.00054.wav':
                    continue

                samples, sr = get_samples(os.path.join(root, file), sample_size)
                specs = get_spectrogram_from_samples(samples, sr)
                dataset.extend(specs)

                ys = [label] * len(samples)
                labels.extend(ys)

        x = standardize_per_sample(torch.from_numpy(np.array(dataset)))
        y = torch.tensor(labels, dtype=torch.long)
        torch.save(x, x_path)
        torch.save(y, y_path)
    else:
        x = torch.load(x_path)
        y = torch.load(y_path)

    print('loaded tensor x:', x.shape)
    print('loaded tensor y', y.shape)
    return train_test_split(x, y, train_size=train_split)


def load_full_cnn_data():
    pass


def standardize_per_sample(x: torch.Tensor):
    mean = x.mean(dim=[2, 3], keepdim=True)
    std = x.std(dim=[2, 3], keepdim=True)
    return (x - mean) / (std + 1e-6)


def get_samples(
    file_path: str,
    sample_len: int,
    stride: int = 1,
    frequency: int | None = None
) -> tuple[list[np.ndarray], int | float]:
    """
    Samples an audio file using a sliding window

    Args:
        file_path (str): the file to sample from in .wav format
        sample_len (int): the length of each sample in seconds
        stride (int): the sample stride in seconds
        frequency (int or None): the audio's sample rate

    Returns:
        samples (list[ndarray]): the list of audio samples
    """
    y, sr = librosa.load(file_path, sr=frequency)

    window_len = int(sample_len * sr)
    stride = int(stride * sr)
    out_len = math.floor((len(y) - window_len) / stride) + 1

    return [y[i * stride : i * stride + window_len] for i in range(out_len)], sr


def get_spectrogram_from_samples(samples, sr):
    spectrograms = []

    for y in samples:
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        spec = spec[:, :spec_dim] # limit to a 128x128 image
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_db = spec_db.reshape(1, spec_dim, spec_dim) # reshape to 1x128x128
        spectrograms.append(spec_db)

    return spectrograms


def plot_samples(
    samples: list[np.ndarray],
    rows: int,
    cols: int,
    frequency: int | None = None
) -> None:
    """
    Plots a list of samples using ``matplotlib`` subplots

    Args:
        samples (list[ndarray]): the list containing all samples
        rows (int): the number of rows in the final plot
        cols (int): the number of columns in the final plot
        frequency (int or None): the sample rate for all samples in the list
    """
    if frequency is None:
        for i, chunk in enumerate(samples):
            plt.subplot(cols, rows, i + 1)
            librosa.display.waveshow(chunk)
    else:
        for i, chunk in enumerate(samples):
            plt.subplot(cols, rows, i + 1)
            librosa.display.waveshow(chunk, sr=frequency)
    plt.show()


def visualize_features(file_path: str, frequency: int | None = None) -> None:
    """
    Visualizes six different audio features for ``file_path`` in .wav format

    waveform, spectogram, mel-spectrogram, mfccs, chroma, and spectral centroid

    Args:
        file_path (str): the path to the file
        frequency (int or None): the audio's sample rate
    """
    y, sr = librosa.load(file_path, sr=frequency)

    # create subplots
    plt.figure(figsize=(14, 8))

    # 1. Waveform
    plt.subplot(6, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 2. Spectrogram (dB)
    plt.subplot(6, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.title('Spectrogram (dB)')
    plt.colorbar(format='%+2.0f dB')

    # 3. Mel-Spectrogram
    plt.subplot(6, 1, 3)
    S = librosa.feature.melspectrogram(y=y[:int(3*sr)], sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel-Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    # 4. MFCCs
    plt.subplot(6, 1, 4)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.title('MFCCs')
    plt.ylabel('MFCC Coefficients')
    plt.colorbar()

    # 5. Chroma
    plt.subplot(6, 1, 5)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
    plt.title('Chroma Features')
    plt.colorbar()

    # 6. Spectral Centroid
    plt.subplot(6, 1, 6)
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(centroids))
    t = librosa.frames_to_time(frames, sr=sr)
    librosa.display.waveshow(y, sr=sr, alpha=0.4)
    plt.plot(t, centroids, color='r')
    plt.title('Spectral Centroid over Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Hz')

    plt.tight_layout()
    plt.show()


def visualize_waveform(file_path: str, color: str):
    genre = re.search('data/genres/(.+)/.*', file_path)
    if genre is None:
        return

    y, sr = librosa.load(file_path)
    plt.figure(figsize=(8, 8))
    librosa.display.waveshow(y, sr=sr, color=color)
    plt.title(f'Waveform of {genre[1]} music')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def visualize_spectogram(file_path: str):
    pass


def visualize_mel_spectogram(file_path: str):
    pass


def visualize_mfcc(file_path: str):
    pass


def visualize_chroma(file_path: str):
    pass


def visualize_spectral_centroid(file_path: str):
    pass


def generate_figure():
    files = [
        'data/genres/country/country.00009.wav',
        'data/genres/pop/pop.00034.wav',
        'data/genres/reggae/reggae.00012.wav',
        'data/genres/classical/classical.00081.wav',
        'data/genres/disco/disco.00001.wav',
        'data/genres/rock/rock.00056.wav'
    ]
    colors = [
        "#DE923A",
        "#33b8c5",
        '#cd5656',
        "#ad2967",
        "#178961",
        "#3731eb"
    ]
    plt.figure(figsize=(10, 8))
    for i in range(len(files)):
        plt.subplot(3, 2, i + 1)

        genre = re.search('data/genres/(.+)/.*', files[i])
        if genre is None:
            continue

        y, sr = librosa.load(files[i])
        librosa.display.waveshow(y, sr=sr, color=colors[i], alpha=0.9)
        plt.title(f'Waveform of {genre[1]} music')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize wave file audio features')
    parser.add_argument('file_paths', type=str, nargs='+', help='Paths to the wave files')
    parser.add_argument(
        '-f', '--features',
        choices=['all', 'wav', 'spec', 'mel', 'mfcc', 'chroma', 'centr'],
        default='all',
        help='Which audio features to visualize'
    )
    parser.add_argument('-c', '--color', type=str, default='1f77b4', help='The color to use when plotting in hex')

    args = parser.parse_args()
    match args.features:
        case 'wav':
            for file_path in args.file_paths:
                visualize_waveform(file_path, '#' + args.color)
        case 'spec':
            pass
        case 'mel':
            pass
        case 'mfcc':
            pass
        case 'chroma':
            pass
        case 'centr':
            pass
        case _:
            generate_figure()


if __name__ == '__main__':
    load_mlp_data()
    load_sampled_cnn_data()

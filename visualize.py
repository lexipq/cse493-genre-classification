import argparse
import librosa
import librosa.display
import math
import matplotlib.pyplot as plt
import numpy as np
import pygame
import re


def init_mixer(frequency: int = 44100, from_file: bool = True) -> None:
    """
    Initializes the pygame mixer for audio playback

    Args:
        frequency (int):
            The frequency in Hz to initialize the mixer with. If not specified,
            defaults to 44100 (44.1 kHz)
        from_file (bool):
            Indicates whether the audio was read from a file path (True)
            or sampled using librosa (False)
    """
    if from_file:
        pygame.mixer.init()
    else:
        pygame.mixer.init(frequency, channels=1)


def play_file_audio(file_path: str) -> None:
    """
    Plays audio from a file using ``pygame.mixer``

    Args:
        file_path (str): the path to the file being played

    .. note:: the pygame mixer must be initialized by calling ``init_mixer()``
    """
    sound = pygame.mixer.Sound(file_path)
    sound.play()


def play_sampled_audio(sample: np.ndarray) -> None:
    """
    Plays the audio sampled using librosa

    Args:
        sample (ndarray): the sample data

    .. note:: the pygame mixer must be initialized by calling ``init_mixer()``
    """
    y_int16 = np.int16(sample * 32767)
    sound_buffer = y_int16.tobytes()

    sound = pygame.mixer.Sound(buffer=sound_buffer)
    sound.play()


def get_samples(
    file_path: str,
    sample_len: int,
    stride: int = 1,
    frequency: int | None = None
) -> list[np.ndarray]:
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

    return [y[i * stride : i * stride + window_len] for i in range(out_len)]


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
            pass


if __name__ == '__main__':
    main()

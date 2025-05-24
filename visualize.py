import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math
import pygame

def init_mixer(frequency: int | None = None, from_file=True):
    if from_file:
        pygame.mixer.init()
    else:
        frequency = 44100 if frequency is None else frequency
        pygame.mixer.init(frequency, channels=1)

def play_audio(file_path):
    sound = pygame.mixer.Sound(file_path)
    sound.play()

def play_sample(sample):
    y_int16 = np.int16(sample * 32767)
    sound_buffer = y_int16.tobytes()

    sound = pygame.mixer.Sound(buffer=sound_buffer)
    sound.play()

def get_samples(file_path: str, sample_len: int, stride=1, frequency: int | None = None):
    """
    Parameters
    ----------
    file_path : the .wav file that you want to sample from
    sample_len : the length of each sample, in seconds
    stride : the stride for each sample window, in seconds
    frequency : the sample rate, defaults to file sample rate

    Returns
    -------
    samples : the list of samples where each entry is a numpy array
    """
    y, sr = librosa.load(file_path, sr=frequency)

    window_len = int(sample_len * sr)
    stride = int(stride * sr)
    out_len = math.floor((len(y) - window_len) / stride) + 1

    return [y[i * stride:i * stride + window_len] for i in range(out_len)]

def plot_samples(samples, frequency, rows, cols):
    """
    Creates subplots using ``matplotlib`` with `cols` number of columns
    and ``rows`` number of rows.

    Parameters
    ----------
    samples : the list containing all samples you want to plot
    frequency : the original sample rate for all samples in the list
    rows : the number of rows in the final plot
    cols : the number of columns in the final plot
    """
    for i, chunk in enumerate(samples):
        plt.subplot(cols, rows, i + 1)
        librosa.display.waveshow(chunk, sr=frequency)

def visualize_features(file_path: str, frequency: int | None = None):
    """
    Visualize 6 different audio features for ``file_path`` in .wav format

    waveform, spectogram, mel-spectrogram, mfccs, chroma, and spectral centroid

    Parameters
    ----------
    file_path : the file to visualize
    frequency : the sample rate, if not specified it uses the file's default rate
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

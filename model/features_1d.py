import math
import librosa
import numpy as np
from numpy import ndarray

def repeat_audio(data: ndarray, sr: int, duration_millis: int):

    length = int(duration_millis/1000*sr)
    n = math.ceil(duration_millis/1000*sr/len(data))
    fixed = np.tile(data, n)
    fixed = fixed[:length]

    return fixed


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


def invert_polarity(data):
    return data * -1


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)


def extract_features_1d(data, sr=22050):
    result = np.array([])

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(
        y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def extract_features_1d_v2(data, sr=22050):
    result = np.array([])

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sr), axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally
    print(chroma_stft.shape)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr), axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally
    print(mfcc.shape)

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(
        y=data, sr=sr), axis=0)
    result = np.hstack((result, mel))  # stacking horizontally
    print(mel.shape)

    return result


def get_features_1d(path, augment=True):

    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path)

    # without augmentation
    res1 = extract_features_1d(data)
    result = np.array(res1)

    if (augment):
        noise_data = noise(data)
        res2 = extract_features_1d(noise_data)
        result = np.vstack((result, res2))  # stacking vertically

        invert_data = invert_polarity(data)
        res3 = extract_features_1d(invert_data)
        result = np.vstack((result, res3))  # stacking vertically
        return result

    result = np.reshape(result, (1, len(result)))
    return result


def extraxt_avg_mfccs(data, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    return mfccs


def get_raw_mfcc(path, fixed_length = -1):
    data, sr = librosa.load(path)
    if (fixed_length > 0):
        data = librosa.util.fix_length(data, size=fixed_length)
    # without augmentation
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return mfcc

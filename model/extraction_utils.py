import pandas as pd
import numpy as np

import librosa

from sklearn.preprocessing import StandardScaler, OneHotEncoder


def __noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


def __invert_polarity(data):
    return data * -1


def __stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)


def __shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)


def __pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def __get_mel_flat(data, sample_rate, n_mels=64, n_fft=1024, hop_length=690):
    mel = librosa.feature.melspectrogram(
        y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    print(mel.shape)
    return mel.flatten()


def __get_mfcc_flat(data, sample_rate, n_mfcc=64, n_fft=1024, hop_length=690):
    mfcc = librosa.feature.mfcc(
        y=data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    print(mfcc.shape)
    return mfcc.flatten()


def __get_stft_flat(data, sample_rate,  n_fft=126, hop_length=690, win_length=126):
    stft = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    print(stft_db.shape)
    return stft_db.flatten()


def __get_features_mel(data, sample_rate, n_mels, n_fft, hop_length, augment):
    res1 = __get_mel_flat(data, sample_rate, n_mels, n_fft, hop_length)
    result = np.array(res1)
    if (augment):
        noise_data = __noise(data)
        res1 = __get_mel_flat(noise_data, sample_rate, n_mels, n_fft, hop_length)
        result = np.vstack((result, res1))

        # invert_data = __invert_polarity(data)
        # res1 = __get_mel_flat(invert_data, sample_rate, n_mels, n_fft, hop_length)
        # result = np.vstack((result, res1))

        shifted_data = __shift(data)
        res1 = __get_mel_flat(shifted_data, sample_rate, n_mels, n_fft, hop_length)
        result = np.vstack((result, res1))

        return result

    result = np.reshape(result, (1, len(result)))
    return result


def __get_features_mfcc(data, sample_rate, n_mfcc, n_fft, hop_length, augment):
    res1 = __get_mfcc_flat(data, sample_rate, n_mfcc, n_fft, hop_length)
    result = np.array(res1)
    if (augment):
        noise_data = __noise(data)
        res1 = __get_mfcc_flat(noise_data, sample_rate, n_mfcc, n_fft, hop_length)
        result = np.vstack((result, res1))

        # invert_data = __invert_polarity(data)
        # res1 = __get_mfcc_flat(invert_data, sample_rate, n_mfcc, n_fft, hop_length)
        # result = np.vstack((result, res1))

        shifted_data = __shift(data)
        res1 = __get_mfcc_flat(shifted_data, sample_rate, n_mfcc, n_fft, hop_length)
        result = np.vstack((result, res1))

        return result

    result = np.reshape(result, (1, len(result)))
    return result


def __get_features_stft(data, sample_rate,n_fft, hop_length, win_length, augment):
    res1 = __get_stft_flat(data, sample_rate, n_fft=n_fft, win_length=win_length,  hop_length=hop_length)
    result = np.array(res1)
    if (augment):
        noise_data = __noise(data)
        res1 = __get_stft_flat(noise_data, sample_rate, n_fft=n_fft, win_length=win_length,  hop_length=hop_length)
        result = np.vstack((result, res1))

        # invert_data = __invert_polarity(data)
        # res1 = __get_stft_flat(invert_data, sample_rate, n_fft=n_fft, win_length=win_length,  hop_length=hop_length)
        # result = np.vstack((result, res1))

        shifted_data = __shift(data)
        res1 = __get_stft_flat(shifted_data, sample_rate, n_fft=n_fft, win_length=win_length,  hop_length=hop_length)
        result = np.vstack((result, res1))

        return result

    result = np.reshape(result, (1, len(result)))
    return result


def __extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def __get_features(data, sample_rate, augment):
    # without augmentation
    res1 = __extract_features(data, sample_rate)
    result = np.array(res1)
    if augment:
        # data with noise
        noise_data = __noise(data)
        res2 = __extract_features(noise_data, sample_rate)
        result = np.vstack((result, res2)) # stacking vertically
        
        # data with stretching and pitching
        shifted_data = __shift(data)
        res3 = __extract_features(shifted_data, sample_rate)
        result = np.vstack((result, res3)) # stacking vertically
        
        return result
    
    result = np.reshape(result, (1, len(result)))
    return result

def extract_v1(data_arr, labels_arr, path, sample_rate=22050, augment=True, shuffle=True):
    X, y = [], []
    for data, label in zip(data_arr, labels_arr):
        features = __get_features(data, sample_rate, augment)
        for f in features:
            X.append(f)
            y.append(label)

    res_df = pd.DataFrame(X)
    res_df['labels'] = y
    
    if shuffle:
        res_df = res_df.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle dataframe

    res_df.to_csv(path, index=False)



def extract_mels(data_arr, labels_arr, path, sample_rate=22050, n_mels=64, n_fft=1024, hop_length=690, augment=True, shuffle=True):
    X, y = [], []
    for data, label in zip(data_arr, labels_arr):
        features = __get_features_mel(data, sample_rate, n_mels, n_fft, hop_length, augment)
        for f in features:
            X.append(f)
            y.append(label)

    res_df = pd.DataFrame(X)
    res_df['labels'] = y
    
    if shuffle:
        res_df = res_df.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle dataframe

    res_df.to_csv(path, index=False)


def extract_mfccs(data_arr, labels_arr, path, sample_rate=22050, n_mfcc=64, n_fft=1024, hop_length=690, augment=True, shuffle=True):
    X, y = [], []
    for data, label in zip(data_arr, labels_arr):
        features = __get_features_mfcc(data, sample_rate, n_mfcc, n_fft, hop_length, augment)
        for f in features:
            X.append(f)
            y.append(label)

    res_df = pd.DataFrame(X)
    res_df['labels'] = y
    
    if shuffle:
        res_df = res_df.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle dataframe

    res_df.to_csv(path, index=False)


def extract_stfts(data_arr, labels_arr, path, sample_rate=22050, n_fft=126, hop_length=690, win_length=126, augment=True, shuffle=True):
    X, y = [], []
    for data, label in zip(data_arr, labels_arr):
        features = __get_features_stft(data, sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length, augment=augment)
        for f in features:
            X.append(f)
            y.append(label)

    res_df = pd.DataFrame(X)
    res_df['labels'] = y
    
    if shuffle:
        res_df = res_df.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle dataframe

    res_df.to_csv(path, index=False)
    
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflow_io as tfio
import pathlib
import os
import pandas as pd
import numpy as np
import librosa as l


from tensorflow.keras.utils import Sequence


class DataLoaderT(Sequence):
    def __init__(self, df: pd.DataFrame, x_col: str, y_col: str, batch_size: int, input_size=(128, 128, 1), shuffle=True):
        self.df = df.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.df)

        self.encoder = OneHotEncoder()

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, idx):

        batches = self.df[idx * self.batch_size:(idx + 1) * self.batch_size]

        x, y = self.__get_data(batches)

        return x, y

    def __get_data(self, batches_):

        path_batch = batches_[self.x_col]
        emotion_batch = batches_[self.y_col]

        x = np.asarray([self.__extract_feature(path) for path in path_batch])

        y = self.encoder.fit_transform(
            np.array(emotion_batch).reshape(-1, 1)).toarray()

        return x, y

    def __extract_feature(self, path):
        self.__load_audio(path, limit=2)
        mel = self.__generate_mel(
            n_fft=1024, hop_length=345, n_mels=128, window=512)
        return mel

    def __generate_mel(self, n_fft=1024, hop_length=345, n_mels=128, window=512):
        spectrogram = tfio.audio.spectrogram(
            self.wav, nfft=n_fft, window=window, stride=hop_length)

        mel_spectrogram = tfio.audio.melscale(
            spectrogram, rate=22050, mels=n_mels, fmin=0, fmax=8000)

        dbscale_mel_spectrogram = tfio.audio.dbscale(
            mel_spectrogram, top_db=80)

        return dbscale_mel_spectrogram

    def __load_audio(self, path, limit=2, sr=22050):
        file_contents = tf.io.read_file(path)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)

        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=sr)
        # division scales the audio values to be between -1.0 and 1.0
        wav = tf.cast(wav, tf.float32) / 32768.0

        wav = self.__fix_length(wav, limit, sr)
        self.wav = wav

    def __fix_length(self, wav, target_duration_sec=2, sr=22050):
        audio_length = len(wav.numpy())
        target_length = target_duration_sec * sr
        if audio_length < target_length:
            # Audio is shorter than target, pad with silence
            padding_amount = tf.cast(
                (target_length - audio_length) / 2, dtype=tf.int32)
            padding = tf.zeros_like(wav[:padding_amount])
            wav = tf.concat([padding, wav, padding], axis=0)
        elif audio_length > target_length:
            # Audio is longer than target, trim from the beginning and end
            start_trim = tf.cast(tf.random.uniform(
                [], minval=0, maxval=audio_length - target_length, dtype=tf.float32), dtype=tf.int32)
            end_trim = start_trim + tf.cast(target_length, dtype=tf.int32)
            wav = wav[start_trim:end_trim]
        return wav

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_input_shape(self):
        mel = self.__extract_feature(self.df.iat[0, 1])
        return mel.shape

    def generate_dataset(self):
        pass            


# dataframe_path = r'D:\Programming\nn\emotion_classification\datasets\combined_no_uk_df.csv'
# df = pd.read_csv(dataframe_path)
# loader = DataLoaderT(df, 'path', 'emotion', 1)
# x, y = loader.__getitem__(0)

# print('x:\n', x.shape)

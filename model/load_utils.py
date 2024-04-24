import pandas as pd
import numpy as np
import librosa


def load_and_scale(df: pd.DataFrame, fixed_length: int, scale=True):
    data = []
    labels = []
    a = True
    for _, row in df.iterrows():
        audio, sample_rate = librosa.load(row['path'])
        if a:
            print(sample_rate)
            a = False
        audio = librosa.util.fix_length(audio, size=fixed_length)
        if scale:
            audio = audio / 32768.0
        data.append(audio)
        labels.append(row['emotion'])

    data = np.array(data)
    labels = np.array(labels)

    return [data, labels]


def load_encode(data_dirs, encoder):

    result = []
    for dir_path in data_dirs:
        df = pd.read_csv(dir_path)

        x = df.iloc[:, :-1].values
        y = df['labels'].values

        x = np.reshape(x, (len(x), 64, 64))
        y_enc = encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()

        print(dir_path)
        print(x.shape, y_enc.shape)

        result.append((x, y_enc))
    result.append((encoder.inverse_transform(result[2][1])))
    return result


def load_encode_single(data_dir, encoder):

    df = pd.read_csv(data_dir)

    x = df.iloc[:, :-1].values
    y = df['labels'].values

    x = np.reshape(x, (len(x), 64, 64))
    y_enc = encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()

    print(data_dir)
    print(x.shape, y_enc.shape)

    return [x, y_enc]


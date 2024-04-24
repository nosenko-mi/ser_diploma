from os import listdir
from os.path import isfile, join, basename
import os
import pandas as pd


fear_label = 'fear'
contempt_label = 'contempt'  # disgust
happiness_label = 'happiness'
anger_label = 'anger'
sadness_label = 'sadness'
neutral_label = 'neutral'
surprise_label = 'surprise'
calm_label = 'calm'

train_uk_data_path = os.fsdecode(
    "D:/Programming/nn/emotion_classification/datasets/emofilm/wav_corpus_uk_new")
train_uk_data_2_path = os.fsdecode("I:\obs\wav_uk_new_train_ext2")
test_uk_data_path = os.fsdecode("I:\Datasets\wav_corpus_uk_test")


eng_emotions = {
    'fea': fear_label,
    'con': contempt_label,
    'hap': happiness_label,
    'ang': anger_label,
    'sad': sadness_label,
    'sur': surprise_label,
    'cal': calm_label
}


emofilm_emotions = {
    'ans': fear_label,
    'dis': contempt_label,
    'gio': happiness_label,
    'rab': anger_label,
    'tri': sadness_label,
    'sur': surprise_label,
    'cal': calm_label
}


def load(path, emotions):

    emofilm_map = {'emotion': [], 'path': []}

    for filename in listdir(path):
        emotion = filename.split('_')[0]
        emofilm_map['emotion'].append(emotions.get(emotion))
        emofilm_map['path'].append(join(path, filename))

    emofilm_df = pd.DataFrame(emofilm_map)

    return emofilm_df

def load_g_first(path, emotions):

    emofilm_map = {'emotion': [], 'path': []}

    for filename in listdir(path):
        emotion = filename.split('_')[1]
        emofilm_map['emotion'].append(emotions.get(emotion))
        emofilm_map['path'].append(join(path, filename))

    emofilm_df = pd.DataFrame(emofilm_map)

    return emofilm_df


train_ext_1 = load("I:\obs\wav_uk_train_ext1", emofilm_emotions)
train_ext_2= load("I:\obs\wav_uk_train_ext2", eng_emotions)
# train_ext_3= load_g_first(r"I:\obs\t", eng_emotions)
test_ext_1= load("I:\obs\wav_uk_test_ext1", emofilm_emotions)

train = pd.concat([train_ext_1, train_ext_2])
test= pd.concat([test_ext_1])

print(train.emotion.value_counts())
print(train.head())

train.to_csv('emofilm_uk_new_df.csv', index=False)
test.to_csv('emofilm_uk_new_test_df.csv', index=False)

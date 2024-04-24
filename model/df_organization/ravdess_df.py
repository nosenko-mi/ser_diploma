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


def load_ravdess(path):
    ravdess_directory_list = os.listdir(path)
    ravdess_emotions_map = {
        1: neutral_label,
        2: calm_label,
        3: happiness_label,
        4: sadness_label,
        5: anger_label,
        6: fear_label,
        7: contempt_label,
        8: surprise_label,
    }

    ravdess_map = {'emotion': [], 'path': [], }
    for dir in ravdess_directory_list:
        # as their are 20 different actors in our previous directory we need to extract files for each actor.
        actors = os.listdir(path + dir)
        for file in actors:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            ravdess_map['emotion'].append(
                ravdess_emotions_map.get(int(part[2])))
            ravdess_map['path'].append(path + dir + '/' + file)

    # dataframe for emotion of files
    ravdess_df = pd.DataFrame(ravdess_map)
    ravdess_df = pd.DataFrame(ravdess_map)

    # changing integers to actual emotions.
    # ravdess_df.emotion.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    #                             5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)

    return ravdess_df


df = load_ravdess(
    path="D:/Programming/nn/emotion_classification/datasets/ravdess_full/")
df.to_csv("ravdess_full_df.csv", index=False)
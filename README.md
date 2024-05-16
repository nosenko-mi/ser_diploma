# Speech Emotion Recognition

Це перша частина  [цієї роботи](https://github.com/nosenko-mi/EmotionClassification).

У цьому репозиторії розглядається архітектура нейронної мережі типу CNN_GRU (GRU шар замісь LSTM було обрано для зменшення складності моделі) для розпізнавання емоцій за аудіо.

df_organization: для кожного набору даних було створено датафрейм "емоція | шлях до файлу". Потім ці дані були використані для генерування мел-спектрограми розміром 64х64 значення.

selected_models: збережені моделі

## Важливі файли

Скрипти:

- ser_diploma\model\feature_extraction_mel.ipynb - виокремлення ознак з аудіо
- ser_diploma\model\cnn_gru.ipynb - скрипт створення останньої моделі

Утиліти:

- ser_diploma\model\save_utils.py - зберігання моделей з метаданими
- ser_diploma\model\metrics_utils.py - зберігання показників моделей
- ser_diploma\model\load_utils.py - завантаження ознак
- ser_diploma\model\extraction_utils.py - виокремлення ознак

## Використані набори даних

- TESS <https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess>
- CREMA-D <https://www.kaggle.com/datasets/ejlok1/cremad>
- RAVDESS <https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio>
- SAVEE <https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee>
- EmoFilm <https://zenodo.org/records/1326428>
- СAFE <https://zenodo.org/records/1478765>
- ASVP-ESD <https://www.kaggle.com/datasets/dejolilandry/asvpesdspeech-nonspeech-emotional-utterances>
- MELD <https://affective-meld.github.io/>
- ESD <https://github.com/HLTSingapore/Emotional-Speech-Data>
- Emov-DB <https://github.com/numediart/EmoV-DB>

## Загальні відомості

- Нейронна мережа: Keras/TensorFlow
- Ознаки з аудіо: Librosa
- Датафрейми та масиви: Pandas, NumPy
- Візуалізації: Matplotlib

## ComputerVision
Pack of programms for video recognizing


## [
  {
    "name": "England",
    "code": "GB",
    "emoji": "🇬🇧",
    "unicode": "U+1F1EE U+1F1E9",
    "image": "https://cdn.jsdelivr.net/npm/country-flag-emoji-json@2.0.0/dist/images/GB.svg"
  },
  // More items...
]  ENG


It contains 4 programs with different specializations

To work, you will need: opencv-python, mediapipe, deepface, tf-keras, ultralytics, torch.


# BodyPosition.py

Arranges the "skeleton" for limbs and body, works only for 1 person.

# Emotions.py


Recognizes the emotions of one person in the frame


# PeoplesTrack.py

It tracks people in the frame (including several at the same time), works with two image outputs - on the first a general view with an outline of people, on the second an approximation of the video to the nearest person.


# FullPeoples.py

Recognizes both people and many other objects from the dataset, people are marked on the first screen with green areas, the rest of the objects are purple. Also, the capture of the first person in the frame is displayed on the second screen.


## 🇷🇺 RUS

Содержит 4 программы с разной специализацией

Для работы потребуются: opencv-python, mediapipe, deepface, tf-keras, ultralytics, torch.

# BodyPosition.py 

Расставляет "скелет" для конечностей и тела, работает только для 1 человека.

# Emotions.py

Распознаёт эмоции одного человека в кадре

# PeoplesTrack.py

Отслеживает людей в кадре (включая несколько единовременно), работает двумя выводами изображения - на первом общий вид с обведением людей, на втором приближение видео на ближайшего человека.

# FullPeoples.py

Распознаёт и людей и множество других объектов из датасета, люди помечаются на первом экране зелёными  областями, остальные объекты - сиреневым. Также на второй экран выводит захват первого человека в кадре.

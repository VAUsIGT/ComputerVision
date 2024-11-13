from deepface import DeepFace
import cv2
import mediapipe as mp

# Инициализация MediaPipe для обнаружения лица
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Захват видео с вебкамеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр с камеры.")
        break

    # Преобразование изображения в RGB (MediaPipe и DeepFace работают с RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Определение лица
    result = face_detection.process(rgb_frame)

    # Если лицо обнаружено
    if result.detections:
        try:
            # Распознавание эмоций на изображении
            emotion = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            # Проверим, если результат - это список, то берем первый элемент
            if isinstance(emotion, list):
                emotion = emotion[0]

            # Получаем основное настроение
            mood = emotion['dominant_emotion']

            # Отображение эмоции на видео
            cv2.putText(frame, f"Emotion: {mood}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Ошибка анализа эмоций: {e}")

    # Показ видео в окне
    cv2.imshow("Webcam - Emotion Detection", frame)

    # Закрытие окна при нажатии клавиши 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Завершение работы
cap.release()
cv2.destroyAllWindows()

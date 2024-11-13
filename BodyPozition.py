import cv2
import mediapipe as mp

# Инициализация MediaPipe для обнаружения позы человека
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils  # Для рисования контуров

# Захват видео с вебкамеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр с камеры.")
        break

    # Преобразование изображения в RGB (MediaPipe работает с RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Определение позы
    result = pose.process(rgb_frame)

    # Если в кадре обнаружен человек
    if result.pose_landmarks:
        # Отображаем ключевые точки и скелет человека
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # Показ видео в окне
    cv2.imshow("Webcam - Human Detection", frame)

    # Закрытие окна при нажатии клавиши 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Завершение работы
cap.release()
cv2.destroyAllWindows()

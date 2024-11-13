import cv2
import torch
from ultralytics import YOLO

# Установим количество потоков для CPU
torch.set_num_threads(4)

# Загружаем модель YOLO для распознавания людей и других объектов
model_people = YOLO("yolov5s.pt")  # Модель для распознавания людей и других объектов

# Функция для обнаружения людей и других объектов
def detect_person_and_objects(frame):
    # Детекция людей и других объектов с основной модели
    results_people = model_people(frame)

    height, width, _ = frame.shape

    person_boxes = []  # Список для хранения координат людей
    other_objects = []  # Список для хранения информации о других объектах

    # Перебираем объекты, обнаруженные моделью для людей и других объектов
    for result in results_people:
        for obj in result.boxes:
            class_id = int(obj.cls)
            label = model_people.names[class_id]

            # Определение координат и размеров объектов
            x, y, w, h = map(int, obj.xywh[0])  # Получаем x, y, ширину и высоту

            # Преобразуем координаты из центра (x, y) и размеры (w, h) в верхний левый и нижний правый углы
            x1, y1 = int(x - w / 2), int(y - h / 2)  # Верхний левый угол
            x2, y2 = int(x + w / 2), int(y + h / 2)  # Нижний правый угол

            # Если обнаружен человек, сохраняем его координаты
            if label == "person":
                person_boxes.append((x1, y1, x2, y2))

            # Для всех остальных объектов сохраняем координаты
            if label != "person":
                other_objects.append((x1, y1, x2, y2, label))

    # Возвращаем результат: координаты людей и других объектов
    return person_boxes, other_objects

# Захват видео с камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр с камеры.")
        break

    # Определяем размер кадра
    height, width, _ = frame.shape

    # Обнаруживаем людей и другие объекты
    person_boxes, other_objects = detect_person_and_objects(frame)

    # Создаем копию кадра для полного изображения
    full_frame = frame.copy()

    # Отображаем прямоугольники для всех людей в полном кадре (зелёные)
    for box in person_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зелёный прямоугольник для людей

    # Отображаем прямоугольники для всех других объектов (сиреневые)
    for box in other_objects:
        x1, y1, x2, y2, label = box
        cv2.rectangle(full_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Сиреневый прямоугольник для других объектов
        cv2.putText(full_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # Показываем полное изображение с прямоугольниками вокруг людей и других объектов
    cv2.imshow("Full Frame with People and Other Objects Detection", full_frame)

    # Если обнаружен человек, динамически центрируем кадр
    if person_boxes:
        # Выбираем первого человека (можно улучшить алгоритм для отслеживания нескольких людей)
        x1, y1, x2, y2 = person_boxes[0]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Центр первого человека
        scale_factor = 1.1  # Дополнительный масштаб для удержания всего человека

        # Рассчитываем область для центрирования человека в кадре с небольшим запасом
        half_w, half_h = int((x2 - x1) * scale_factor), int((y2 - y1) * scale_factor)
        top_left_x = max(0, cx - half_w)
        top_left_y = max(0, cy - half_h)
        bottom_right_x = min(width, cx + half_w)
        bottom_right_y = min(height, cy + half_h)

        # Извлекаем область интереса и масштабируем её до размера окна
        zoomed_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        zoomed_frame = cv2.resize(zoomed_frame, (width, height), interpolation=cv2.INTER_LINEAR)

        # Показываем увеличенное изображение с зумом на человеке
        cv2.imshow("Zoomed Frame with Person Detection", zoomed_frame)
    else:
        # Если человек не обнаружен, показываем обычное изображение
        cv2.imshow("Zoomed Frame with Person Detection", frame)

    # Завершаем при нажатии 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Завершение работы
cap.release()
cv2.destroyAllWindows()

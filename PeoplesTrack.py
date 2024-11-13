import cv2
import torch
from ultralytics import YOLO

# Установим количество потоков для CPU
torch.set_num_threads(4)

# Загружаем модель YOLOv5
model = YOLO("yolov5s.pt")  # Выберите модель с хорошими результатами на людях

# Функция для обнаружения людей и одежды
def detect_person_and_clothes(frame):
    results = model(frame)  # Выполняем детекцию
    height, width, _ = frame.shape

    person_boxes = []  # Список для хранения координат людей
    clothes_info = []  # Список для хранения информации о одежде

    # Перебираем объекты, обнаруженные моделью
    for result in results:
        for obj in result.boxes:
            class_id = int(obj.cls)
            label = model.names[class_id]

            # Определение координат и размеров объектов
            x, y, w, h = map(int, obj.xywh[0])  # Получаем x, y, ширину и высоту

            # Преобразуем координаты из центра (x, y) и размеры (w, h) в верхний левый и нижний правый углы
            x1, y1 = int(x - w / 2), int(y - h / 2)  # Верхний левый угол
            x2, y2 = int(x + w / 2), int(y + h / 2)  # Нижний правый угол

            # Если обнаружен человек, сохраняем его координаты
            if label == "person":
                person_boxes.append((x1, y1, x2, y2))

            # Проверка на одежду
            if label in ["shirt", "pants", "dress", "hat", "jacket"]:
                clothes_info.append(label)

    # Возвращаем результат: найден ли человек и информация о его одежде
    return person_boxes, clothes_info

# Захват видео с камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить кадр с камеры.")
        break

    # Определяем размер кадра
    height, width, _ = frame.shape

    # Обнаруживаем людей и их одежду
    person_boxes, clothes_info = detect_person_and_clothes(frame)

    # Отображаем прямоугольники для всех людей в полном кадре
    full_frame = frame.copy()  # Создаем копию кадра для полного изображения

    for box in person_boxes:
        x1, y1, x2, y2 = box
        # Рисуем прямоугольник для каждого человека
        cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Обводим прямоугольником

    # Показываем полное изображение с прямоугольниками вокруг людей
    cv2.imshow("Full Frame with People Detection", full_frame)

    # Если обнаружен человек, динамически центрируем кадр
    if person_boxes:
        # Выбираем первого человека (можно улучшить алгоритм для отслеживания нескольких людей)
        x1, y1, x2, y2 = person_boxes[0]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Центр первого человека
        scale_factor = 1.2  # Дополнительный масштаб для удержания всего человека

        # Рассчитываем область для центрирования человека в кадре с небольшим запасом
        half_w, half_h = int((x2 - x1) * scale_factor), int((y2 - y1) * scale_factor)
        top_left_x = max(0, cx - half_w)
        top_left_y = max(0, cy - half_h)
        bottom_right_x = min(width, cx + half_w)
        bottom_right_y = min(height, cy + half_h)

        # Извлекаем область интереса и масштабируем её до размера окна
        zoomed_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        zoomed_frame = cv2.resize(zoomed_frame, (width, height), interpolation=cv2.INTER_LINEAR)

        # Отображаем информацию об одежде
        clothing_text = ", ".join(clothes_info) if clothes_info else "No clothing detected"
        cv2.putText(zoomed_frame, f"Clothing: {clothing_text}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)

        # Показываем увеличенное изображение с зумом и информацией об одежде
        cv2.imshow("Zoomed Frame with Clothing Detection", zoomed_frame)
    else:
        # Если человек не обнаружен, показываем обычное изображение
        cv2.imshow("Zoomed Frame with Clothing Detection", frame)

    # Завершаем при нажатии 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Завершение работы
cap.release()
cv2.destroyAllWindows()

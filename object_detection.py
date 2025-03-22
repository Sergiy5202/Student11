import cv2
from ultralytics import YOLO
import numpy as np
import time

def init_model(model_path="yolov8n.pt"):
    """Ініціалізація моделі YOLOv8"""
    try:
        model = YOLO(model_path)
        print(f"Модель {model_path} успішно завантажена!")
        return model
    except Exception as e:
        print(f"Помилка при завантаженні моделі: {e}")
        return None

def process_image(model, image_path):
    """Обробка статичного зображення"""
    try:
        # Завантаження зображення
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Не вдалося завантажити зображення")

        # Виконання детекції
        results = model(image)
        
        # Візуалізація результатів
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Отримання координат
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Отримання класу та впевненості
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = result.names[cls]
                
                # Малювання рамки та підпису
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Показ результату
        cv2.imshow("Object Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Помилка при обробці зображення: {e}")

def process_video(model, source=0):
    """Обробка відеопотоку"""
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError("Не вдалося відкрити відеопотік")

        fps = 0
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Виконання детекції
            results = model(frame)

            # Візуалізація результатів
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Отримання координат
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Отримання класу та впевненості
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = result.names[cls]
                    
                    # Малювання рамки та підпису
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Обчислення FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Додавання FPS на кадр
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Показ результату
            cv2.imshow("Object Detection", frame)

            # Вихід при натисканні 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Помилка при обробці відео: {e}")

def main():
    # Ініціалізація моделі
    model = init_model()
    if model is None:
        return

    while True:
        print("\nОберіть режим роботи:")
        print("1. Обробка зображення")
        print("2. Обробка відео з камери")
        print("3. Вихід")

        choice = input("Ваш вибір (1-3): ")

        if choice == '1':
            image_path = input("Введіть шлях до зображення: ")
            process_image(model, image_path)
        elif choice == '2':
            process_video(model)
        elif choice == '3':
            break
        else:
            print("Невірний вибір. Спробуйте ще раз.")

if __name__ == "__main__":
    main()
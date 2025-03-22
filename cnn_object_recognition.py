import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_prepare_mnist_data():
    """
    Завантаження та підготовка датасету MNIST для навчання CNN
    """
    print("Завантаження датасету MNIST...")
    # Завантаження датасету MNIST (зображення рукописних цифр)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Нормалізація та перетворення розміру
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
    
    # Перетворення міток у one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    print(f"Форма тренувальних даних: {X_train.shape}")
    print(f"Форма тестових даних: {X_test.shape}")
    
    return (X_train, y_train), (X_test, y_test)

def create_cnn_model():
    """
    Створення архітектури CNN моделі
    """
    print("Створення CNN моделі...")
    # Побудова CNN-моделі
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # 10 класів (цифри 0-9)
    ])
    
    # Компіляція моделі
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Виведення структури моделі
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
    """
    Навчання CNN моделі
    """
    print(f"Навчання моделі ({epochs} епох)...")
    # Навчання моделі
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Оцінка точності моделі на тестовому наборі
    """
    print("Оцінка моделі на тестових даних...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Точність на тестовому наборі: {test_accuracy:.4f}")
    return test_loss, test_accuracy

def visualize_training_history(history):
    """
    Візуалізація процесу навчання (точність і втрати)
    """
    print("Візуалізація процесу навчання...")
    # Графік точності
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точність на тренувальному наборі')
    plt.plot(history.history['val_accuracy'], label='Точність на валідаційному наборі')
    plt.title('Точність моделі')
    plt.xlabel('Епоха')
    plt.ylabel('Точність')
    plt.legend()
    
    # Графік втрат
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Втрати на тренувальному наборі')
    plt.plot(history.history['val_loss'], label='Втрати на валідаційному наборі')
    plt.title('Втрати моделі')
    plt.xlabel('Епоха')
    plt.ylabel('Втрати')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, X_test, y_test, num_samples=10):
    """
    Візуалізація прикладів передбачень моделі
    """
    print("Візуалізація прикладів передбачень...")
    # Отримання передбачень
    predictions = model.predict(X_test[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:num_samples], axis=1)
    
    # Відображення зображень та передбачень
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
        plt.title(f"Прогноз: {predicted_classes[i]}\nФакт: {true_classes[i]}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_model(model, model_path="cnn_mnist_model"):
    """
    Збереження навченої моделі
    """
    print(f"Збереження моделі в {model_path}...")
    model.save(model_path)
    print(f"Модель збережена в {model_path}")

def load_saved_model(model_path="cnn_mnist_model"):
    """
    Завантаження збереженої моделі
    """
    print(f"Завантаження моделі з {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Модель успішно завантажена!")
    return model

def predict_single_image(model, image_path=None):
    """
    Передбачення класу для одного зображення
    Якщо image_path не вказано, використовується випадкове зображення з тестового набору
    """
    if image_path and os.path.exists(image_path):
        # Завантаження та підготовка зображення
        img = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape(1, 28, 28, 1) / 255.0
    else:
        # Використання випадкового зображення з тестового набору
        print("Шлях до зображення не вказано або файл не існує. Використання випадкового зображення з тестового набору.")
        (_, _), (X_test, y_test) = mnist.load_data()
        idx = np.random.randint(0, len(X_test))
        img_array = X_test[idx].reshape(1, 28, 28, 1) / 255.0
        true_label = y_test[idx]
        
        # Відображення зображення
        plt.figure(figsize=(4, 4))
        plt.imshow(X_test[idx], cmap='gray')
        plt.title(f"Справжній клас: {true_label}")
        plt.axis('off')
        plt.show()
    
    # Отримання передбачення
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print(f"Передбачений клас: {predicted_class}, Впевненість: {confidence:.2f}%")
    return predicted_class, confidence

def main():
    """
    Основна функція для демонстрації роботи CNN
    """
    print("=== Демонстрація згорткової нейронної мережі (CNN) для розпізнавання об'єктів ===")
    
    # Завантаження та підготовка даних
    (X_train, y_train), (X_test, y_test) = load_and_prepare_mnist_data()
    
    # Створення моделі
    model = create_cnn_model()
    
    # Навчання моделі
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=5)
    
    # Оцінка моделі
    evaluate_model(model, X_test, y_test)
    
    # Візуалізація процесу навчання
    visualize_training_history(history)
    
    # Візуалізація передбачень
    visualize_predictions(model, X_test, y_test)
    
    # Збереження моделі
    save_model(model)
    
    print("\nДемонстрація завершена!")

if __name__ == "__main__":
    main()
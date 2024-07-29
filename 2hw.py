import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import requests
from io import BytesIO
import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD

# Вставте URL-адреси зображень
image_urls = [
    "https://www.tailorbrands.com/wp-content/uploads/2020/07/apple-logo.jpg",
    "https://www.tailorbrands.com/wp-content/uploads/2020/07/mcdonalds-logo.jpg",
    "https://www.tailorbrands.com/wp-content/uploads/2020/07/twitter-logo.jpg",
    # додайте більше URL-адрес за потреби
]

# Виведення та обробка зображень
for url in image_urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = np.array(img)

    # Відображення зображення
    plt.imshow(img)
    plt.axis('off')  # Приховати осі
    plt.title(f'Зображення з URL: {url}')
    plt.show()

    # Визначення розміру зображення
    print(f'Розмір зображення з URL {url}: {img.shape}')

    # змінити форму зображення з 3D-матриці на 2D-матрицю шляхом укладання кольорових каналів горизонтально
    height, width, channels = img.shape
    flat_image = img.reshape(-1, width * channels)

    plt.imshow(flat_image)
    plt.axis('off')  # Приховати осі
    plt.title(f'Зображення в 2D (плоске)')
    plt.show()
    print(f'Розмір зображення в 2D: {flat_image.shape}')


    # Обчислення SVD розкладу
    U, S, Vt = np.linalg.svd(flat_image, full_matrices=False)
    print(f"Shape of U={U.shape}")
    print(f"Shape of S={S.shape}")
    print(f"Shape of Vt={Vt.shape}")

    # Візуалізація сингулярних значень
    k = 10  # можна змінити це значення на будь-яке інше для візуалізації
    plt.plot(np.arange(k), S[:k])
    plt.xlabel('Ранг сингулярного значення')
    plt.ylabel('Величина сингулярного значення')
    plt.title('Сингулярні значення')
    plt.show()

    # Обрізати зображення за допомогою SVD
    nums = [5, 15, 30, 50, 75, 100]
    for num in nums:
        svd = TruncatedSVD(n_components=num)
        truncated_image = svd.fit_transform(flat_image)
        print()
        print('Розмір стиснутого зображення з n-компонентом', num, ': ', truncated_image.shape)

        print('Відновлення . . . .')
        # Відновити зображення зі зменшеного представлення
        reconstructed_image = svd.inverse_transform(truncated_image)

        # розрахунок помилки реконструкції:
        reconstruction_error = np.mean(np.square(reconstructed_image - flat_image))
        print("    Помилка реконструкції", reconstruction_error)

        # Змінити форму зображення до оригінальної 3D форми
        reconstructed_image = reconstructed_image.reshape(height, width, channels)
        
        # Обрізати вихідні дані до цілих чисел у діапазоні [0, 255]
        reconstructed_image = np.clip(reconstructed_image, 0 , 255 ).astype( 'uint8' )
        print('    Розмір відновленого зображення', reconstructed_image.shape)

        # Відображення оригінального та відновленого зображень
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].set_title('Оригінальне зображення')
        axes[0].axis('off')

        axes[1].imshow(reconstructed_image)
        axes[1].set_title(f'Якість відновлення: {num} компонентів')
        axes[1].axis('off')

        plt.show()

# Збереження зображення на жорсткий диск (якщо потрібно)
# plt.savefig('reconstructed_image.jpg')
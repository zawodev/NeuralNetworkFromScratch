import numpy as np
import pathlib
from PIL import Image


def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels


def get_myimg(image_path):
    # Wczytanie obrazu
    img = Image.open(image_path)

    # Zmiana rozmiaru obrazu na 28x28
    img_resized = img.resize((28, 28))

    # Konwersja obrazu na skalę szarości
    img_gray = img_resized.convert('L')

    # Konwersja obrazu do tablicy numpy i skalowanie wartości pikseli do przedziału 0-1
    img_array = 1.0 - np.array(img_gray) / 255.0

    return img_array


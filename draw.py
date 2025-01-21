import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw

model = load_model('mnistV2.h5')

def predict_number(image):
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image)
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=(0, -1))
    prediction = model.predict(image)
    return np.argmax(prediction)

class NumberDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dibujar un número")
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button_predict = tk.Button(root, text="Predecir", command=self.predict)
        self.button_predict.pack()
        self.button_clear = tk.Button(root, text="Limpiar", command=self.clear_canvas)
        self.button_clear.pack()
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = event.x - 10, event.y - 10
        x2, y2 = event.x + 10, event.y + 10
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def predict(self):
        image = np.array(self.image)
        number = predict_number(image)
        messagebox.showinfo("Predicción", f"El número predicho es: {number}")


    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

if __name__ == "__main__":
    root = tk.Tk()
    app = NumberDrawApp(root)
    root.mainloop()
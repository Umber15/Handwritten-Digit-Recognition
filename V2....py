import cv2
import numpy as np
import joblib
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.utils import shuffle
import os

# Load Dataset (Sampled)
train = pd.read_csv("C:/Users/Saif/Desktop/mnist_train.csv")
test = pd.read_csv("C:/Users/Saif/Desktop/mnist_test.csv")
train = shuffle(train, random_state=42).head(6000)
test = shuffle(test, random_state=42).head(1000)

X_train = train.drop('label', axis=1).astype('float32') / 255.0
y_train = train['label']
X_test = test.drop('label', axis=1).astype('float32') / 255.0
y_test = test['label']

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
joblib.dump(clf, "model/digit_recognizer.pkl")
model = joblib.load("model/digit_recognizer.pkl")

def preprocess(img):
    img = img.resize((28, 28)).convert('L')
    img_np = np.array(img)
    img_np = cv2.bitwise_not(img_np)
    _, img_np = cv2.threshold(img_np, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = img_np[y:y + h, x:x + w]
    else:
        digit = img_np

    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    return padded.reshape(1, -1).astype('float32') / 255.0, padded

class StylishDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("✒️ Handwritten Digit Recognizer")
        self.root.geometry("960x500")
        self.root.configure(bg="#2c2c2c")

        title = Label(root, text="✒️ Handwritten Digit Recognizer", font=("Segoe UI", 24, "bold"),
                      bg="#1f1f1f", fg="#f1f1f1", pady=15)
        title.pack(fill=X)

        content = Frame(root, bg="#2c2c2c")
        content.pack(pady=20)

        # Drawing Area
        self.canvas_frame = Frame(content, bg="#2c2c2c")
        self.canvas_frame.grid(row=0, column=0, padx=30)

        self.canvas = Canvas(self.canvas_frame, width=300, height=300, bg='white', bd=4, relief=RIDGE, cursor="cross")
        self.canvas.pack()
        self.image1 = Image.new("RGB", (300, 300), "white")
        self.draw = ImageDraw.Draw(self.image1)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        self.btn_predict = Button(self.canvas_frame, text="🎯 Predict", command=self.predict_digit,
                                  font=("Segoe UI", 12, "bold"), bg="#000000", fg="white",
                                  activebackground="#333333", activeforeground="white",
                                  bd=0, padx=30, pady=10, relief=RAISED)
        self.btn_predict.pack(pady=(20, 10))

        self.btn_clear = Button(self.canvas_frame, text="🧹 Clear", command=self.clear_canvas,
                                font=("Segoe UI", 12, "bold"), bg="#444", fg="white",
                                activebackground="#666", activeforeground="white",
                                bd=0, padx=30, pady=10, relief=RAISED)
        self.btn_clear.pack()

        # Prediction Display
        self.result_frame = Frame(content, bg="#2c2c2c")
        self.result_frame.grid(row=0, column=1, padx=30)

        self.result_label = Label(self.result_frame, text="Prediction: ", font=("Segoe UI", 20, "bold"),
                                  bg="#2c2c2c", fg="#00cec9")
        self.result_label.pack()

        self.result_canvas = Canvas(self.result_frame, width=200, height=200, bg='black', bd=3, relief=RIDGE)
        self.result_canvas.pack(pady=10)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 300, 300], fill="white")
        self.result_label.config(text="Prediction: ")
        self.result_canvas.delete("all")

    def predict_digit(self):
        img = self.image1.copy()
        roi_flat, processed = preprocess(img)
        prediction = model.predict(roi_flat)[0]
        self.result_label.config(text=f"Prediction: {prediction}")

        result_img = cv2.resize(processed, (200, 200), interpolation=cv2.INTER_NEAREST)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
        result_pil = Image.fromarray(result_img)
        result_tk = ImageTk.PhotoImage(image=result_pil)
        self.result_canvas.create_image(0, 0, anchor=NW, image=result_tk)
        self.result_canvas.image = result_tk

# Run App
root = Tk()
app = StylishDigitApp(root)
root.mainloop()

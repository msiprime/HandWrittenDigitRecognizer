import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import filedialog
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2
import time

# Load the trained model
model = tf.keras.models.load_model('mnist_model.keras')


# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image


# Function to predict the digit from an image file
def predict_digit(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return np.argmax(prediction)


# Function to open file dialog and predict the digit
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        digit = predict_digit(file_path)
        animate_result(f'Predicted Digit: {digit}')


# Function to preprocess image from webcam
def preprocess_webcam_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = np.invert(image)
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image


# Function to predict digit from webcam image
def predict_digit_from_webcam(image):
    image = preprocess_webcam_image(image)
    prediction = model.predict(image)
    return np.argmax(prediction)


# Function to close webcam
def close_webcam():
    if 'cap' in globals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()


# Function to close the entire application
def close_application():
    close_webcam()
    root.destroy()


# Function to start webcam for digit prediction
global cap


def start_webcam():
    webcam_cap = cv2.VideoCapture(0)

    while True:
        ret, webcam_frame = webcam_cap.read()
        if not ret:
            break

        roi = webcam_frame[100:400, 100:400]
        cv2.rectangle(webcam_frame, (100, 100), (400, 400), (0, 255, 0), 2)

        digit = predict_digit_from_webcam(roi)
        cv2.putText(webcam_frame, f'Predicted Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Webcam Digit Recognition', webcam_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    close_webcam()


# Drawing canvas for drawing digits
class DrawingCanvas(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent, width=200, height=200, bg='white', highlightthickness=0)
        self.old_y = None
        self.old_x = None
        self.bind("<B1-Motion>", self.paint)
        self.bind("<ButtonRelease-1>", self.predict_drawing)
        self.setup()

    def setup(self):
        self.old_x = None
        self.old_y = None

    def paint(self, event):
        if self.old_x and self.old_y:
            self.create_line(self.old_x, self.old_y, event.x, event.y, width=8, fill='black', capstyle=tk.ROUND,
                             smooth=tk.TRUE)
        self.old_x = event.x
        self.old_y = event.y

    def predict_drawing(self, event):
        self.old_x = None
        self.old_y = None
        self.update()
        # Save canvas drawing to an image
        self.postscript(file='drawing.eps')
        image = Image.open('drawing.eps')
        image = image.convert('L')
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        image = image.reshape(1, 28, 28, 1)
        prediction = model.predict(image)
        digit = np.argmax(prediction)
        animate_result(f'Predicted Digit: {digit}')

    def clear_canvas(self):
        self.delete('all')
        result_label.config(text='Predicted Digit: ')


# Function to animate result label
def animate_result(text):
    result_label.config(text=text)
    font_size = 18
    for _ in range(10):
        font_size += 0
        result_label.config(font=('Helvetica', font_size))
        root.update()
        time.sleep(0.05)
    for _ in range(5):
        font_size -= 0
        result_label.config(font=('Helvetica', font_size))
        root.update()
        time.sleep(0.05)


# Function to add hover effect to buttons
def on_enter(event):
    event.widget.config(style="Hover.TButton")


def on_leave(event):
    event.widget.config(style="TButton")


# Create the main window with a modern theme
root = ThemedTk(theme="arc")
root.title('Hand Written Digit Recognizer')

# Ensure application is closed properly
root.protocol("WM_DELETE_WINDOW", close_application)

# Create a frame with ttk theme
frame = ttk.Frame(root, padding=20)
frame.pack()

# Label for university name (larger font)
university_label = ttk.Label(frame,
                             text="Green University of Bangladesh",
                             font=("Arial", 28),
                             foreground="green")  # Set custom color
university_label.pack()

# Label for IDs (smaller font)
id_label = ttk.Label(frame,
                     text="(Id 212002103 & 212002064)",
                     font=("Arial", 15),
                     foreground="gray")  # Set custom color
id_label.pack()

# Create a frame for buttons and result label
button_frame = ttk.Frame(root, padding=20)
button_frame.pack(pady=20)

# Load Image button
load_button = ttk.Button(button_frame, text='Load Image', command=load_image)
load_button.pack(pady=5)
load_button.bind("<Enter>", on_enter)
load_button.bind("<Leave>", on_leave)

# Start Webcam button
webcam_button = ttk.Button(button_frame, text='Start Webcam', command=start_webcam)
webcam_button.pack(pady=5)
webcam_button.bind("<Enter>", on_enter)
webcam_button.bind("<Leave>", on_leave)

# Result label
result_label = ttk.Label(button_frame, text='Predicted Digit: ', font=('Helvetica', 18))
result_label.pack(pady=20)

# Drawing canvas
drawing_canvas = DrawingCanvas(root)
drawing_canvas.pack(pady=10)

# Erase button
erase_button = ttk.Button(button_frame, text='Erase', command=drawing_canvas.clear_canvas)
erase_button.pack(pady=5)
erase_button.bind("<Enter>", on_enter)
erase_button.bind("<Leave>", on_leave)

# # Add custom style for hover effect

style = ttk.Style()
style.configure("Hover.TButton", font=('Helvetica', 12, 'bold'), background='green',
                bordercolor='green', lightcolor='light green', darkcolor='dark green')

# Apply the style to the button
load_button_style = ttk.Style()
load_button_style.map("Hover.TButton",
                      background=[('active', 'green')],
                      foreground=[('active', 'green')])

# Apply the style to the button
load_button_style = ttk.Style()
load_button_style.map("Hover.TButton",
                      background=[('active', 'green')],
                      foreground=[('active', 'green')])

root.mainloop()

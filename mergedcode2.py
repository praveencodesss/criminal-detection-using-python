import cv2
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import tkinter as tk
from tkinter import messagebox

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)

# Load face recognition model and encodings
def load_face_model_and_encodings():
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    encoding_dict = load_pickle(encodings_path)
    return face_encoder, encoding_dict

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


# Function to detect faces and recognize criminals
def detect_faces_and_recognize(frame, face_detector, face_encoder, encoding_dict):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(img_rgb)
    criminal_detected = False
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(face_encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'not a criminal'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist
                criminal_detected = True

        if criminal_detected:
            cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(frame, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            show_criminal_info_window()

        else:
            cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(frame, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return frame

# Function to load encodings from a pickle file
def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

# Function to show Tkinter criminal information form
def show_criminal_info_window():
    # Create the main window
    root = tk.Tk()
    root.title("Criminal Information Form")

    # Create and place the labels and entry widgets
    tk.Label(root, text="Criminal Name:").grid(row=0, column=0, padx=10, pady=5)
    entry_name = tk.Entry(root)
    entry_name.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text="Criminal ID:").grid(row=1, column=0, padx=10, pady=5)
    entry_id = tk.Entry(root)
    entry_id.grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text="Age:").grid(row=2, column=0, padx=10, pady=5)
    entry_age = tk.Entry(root)
    entry_age.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(root, text="Height (m):").grid(row=3, column=0, padx=10, pady=5)
    entry_height = tk.Entry(root)
    entry_height.grid(row=3, column=1, padx=10, pady=5)

    tk.Label(root, text="Weight (kg):").grid(row=4, column=0, padx=10, pady=5)
    entry_weight = tk.Entry(root)
    entry_weight.grid(row=4, column=1, padx=10, pady=5)

    # Create and place the submit button
    submit_button = tk.Button(root, text="Submit", command=lambda: submit(root, entry_name.get(), entry_id.get(), entry_age.get(), entry_height.get(), entry_weight.get()))
    submit_button.grid(row=5, columnspan=2, pady=10)

    # Create and place the label to display the result
    global display_label
    display_label = tk.Label(root, text="")
    display_label.grid(row=6, columnspan=2, pady=10)

    root.mainloop()

# Function to handle submit button in Tkinter window
def submit(root, name, criminal_id, age, height, weight):
    if not (name and criminal_id and age and height and weight):
        messagebox.showwarning("Input Error", "All fields are required!")
        return

    try:
        age = int(age)
        height = float(height)
        weight = float(weight)
    except ValueError:
        messagebox.showerror("Input Error", "Age must be an integer and height and weight must be numbers!")
        return

    display_text = f"Name: {name}\nCriminal ID: {criminal_id}\nAge: {age}\nHeight: {height} m\nWeight: {weight} kg"
    display_label.config(text=display_text)
    root.destroy()

# Main function
if __name__ == "__main__":
    # Initialize face detection components
    face_detector = mtcnn.MTCNN()
    face_encoder, encoding_dict = load_face_model_and_encodings()

    # Open webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPENED")
            break

        frame = detect_faces_and_recognize(frame, face_detector, face_encoder, encoding_dict)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

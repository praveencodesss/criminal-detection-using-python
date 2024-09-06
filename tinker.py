import tkinter as tk
from tkinter import filedialog
import pandas as pd
import cv2
from PIL import Image, ImageTk
import numpy as np
import mtcnn
from scipy.spatial.distance import cosine
from architecture import InceptionResNetV2
from train_v2 import normalize, l2_normalizer
import pickle

# Constants for face recognition
confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)

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

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img, detector, encoder, encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    detected_names = []
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'not a criminal'
        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name != 'not a criminal':
            detected_names.append((name, distance, pt_1, pt_2))
        cv2.rectangle(img, pt_1, pt_2, (0, 255, 0) if name != 'not a criminal' else (0, 0, 255), 2)
        cv2.putText(img, name + (f'__{distance:.2f}' if name != 'not a criminal' else ''), (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
    
    return img, detected_names

class CriminalInfoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Criminal Information Form")

        # Define the path to the Excel file
        self.excel_file_path = "C:/Users/samay/Downloads/Face-recognition-Using-Facenet-On-Tensorflow-2.X-master/extracted/Face-recognition-Using-Facenet-On-Tensorflow-2.X-master/criminal data.xlsx"

        # Load face recognition model and data
        self.face_encoder = InceptionResNetV2()
        self.face_encoder.load_weights("facenet_keras_weights.h5")
        self.encoding_dict = load_pickle('encodings/encodings.pkl')
        self.face_detector = mtcnn.MTCNN()

        # Create and place the labels and entry widgets
        self.create_widgets()

        # Fetch and populate data from the Excel file
        self.data = self.fetch_data_from_excel()

        # Start video capture
        self.cap = cv2.VideoCapture(0)

        # Schedule the update function
        self.update_video()

    def create_widgets(self):
        tk.Label(self.root, text="Criminal Name:").grid(row=0, column=0, padx=10, pady=5)
        self.entry_name = tk.Entry(self.root)
        self.entry_name.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Criminal ID:").grid(row=1, column=0, padx=10, pady=5)
        self.entry_id = tk.Entry(self.root)
        self.entry_id.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Crime Type:").grid(row=2, column=0, padx=10, pady=5)
        self.entry_crime_type = tk.Entry(self.root)
        self.entry_crime_type.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Age:").grid(row=3, column=0, padx=10, pady=5)
        self.entry_age = tk.Entry(self.root)
        self.entry_age.grid(row=3, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Height (m):").grid(row=4, column=0, padx=10, pady=5)
        self.entry_height = tk.Entry(self.root)
        self.entry_height.grid(row=4, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Weight (kg):").grid(row=5, column=0, padx=10, pady=5)
        self.entry_weight = tk.Entry(self.root)
        self.entry_weight.grid(row=5, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Occupation:").grid(row=6, column=0, padx=10, pady=5)
        self.entry_occupation = tk.Entry(self.root)
        self.entry_occupation.grid(row=6, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Address:").grid(row=7, column=0, padx=10, pady=5)
        self.entry_address = tk.Entry(self.root)
        self.entry_address.grid(row=7, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Arrest Type:").grid(row=8, column=0, padx=10, pady=5)
        self.entry_arrest_type = tk.Entry(self.root)
        self.entry_arrest_type.grid(row=8, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Gender:").grid(row=9, column=0, padx=10, pady=5)
        self.entry_gender = tk.Entry(self.root)
        self.entry_gender.grid(row=9, column=1, padx=10, pady=5)

        # Create and place the label to display the result
        self.display_label = tk.Label(self.root, text="", justify="left")
        self.display_label.grid(row=10, columnspan=2, pady=10)

        # Label to show the image
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=11, columnspan=2, pady=10)

    def fetch_data_from_excel(self):
        # Read the Excel file
        try:
            df = pd.read_excel(self.excel_file_path)
            return df
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return pd.DataFrame()

    def populate_form(self, data_row):
        # Populate the form with data from the selected row of the Excel file
        self.entry_name.delete(0, tk.END)
        self.entry_name.insert(0, data_row.get('Criminal Name', ''))

        self.entry_id.delete(0, tk.END)
        self.entry_id.insert(0, data_row.get('Criminal ID', ''))

        self.entry_crime_type.delete(0, tk.END)
        self.entry_crime_type.insert(0, data_row.get('Crime Type', ''))

        self.entry_age.delete(0, tk.END)
        self.entry_age.insert(0, data_row.get('Age', ''))

        self.entry_height.delete(0, tk.END)
        self.entry_height.insert(0, data_row.get('Height (m)', ''))

        self.entry_weight.delete(0, tk.END)
        self.entry_weight.insert(0, data_row.get('Weight (kg)', ''))

        self.entry_occupation.delete(0, tk.END)
        self.entry_occupation.insert(0, data_row.get('Occupation', ''))

        self.entry_address.delete(0, tk.END)
        self.entry_address.insert(0, data_row.get('Address', ''))

        self.entry_arrest_type.delete(0, tk.END)
        self.entry_arrest_type.insert(0, data_row.get('Arrest Type', ''))

        self.entry_gender.delete(0, tk.END)
        self.entry_gender.insert(0, data_row.get('Gender', ''))

    def update_video(self):
        # Capture video from the camera
        ret, frame = self.cap.read()
        if ret:
            # Perform face detection and recognition
            frame, detected_names = detect(frame, self.face_detector, self.face_encoder, self.encoding_dict)
            
            # Update the GUI with detected names and relevant information
            if detected

import face_recognition
import cv2
import numpy as np
from tkinter import *
import os
import pickle
import threading

data_path = 'images_data'
model_path = 'trained_model.pkl'

root = Tk()

name_var = StringVar()
capturing = False  # Variable para indicar si se está capturando o no
cam = cv2.VideoCapture(0)  # Inicializar la cámara
thread = None  # Hilo para la captura de video

def take_picture(name):
    person_folder = os.path.join(data_path, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    ret, frame = cam.read()
    img_name = os.path.join(person_folder, f"{name}_img.jpg")
    cv2.imwrite(img_name, frame)
    print(f"Image saved for {name}!")

def start_capture():
    global capturing
    name = name_var.get().strip()
    if not name:
        print("Por favor, ingrese un nombre antes de capturar imágenes.")
        return

    while capturing:
        ret, frame = cam.read()

        # Mostrar el video en la ventana
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):  # Presionar 'c' para capturar
            take_picture(name)

    # Restaurar el video al cerrar la captura
    cv2.imshow('Video', frame)

def capture_video():
    global capturing, thread
    if thread is None or not thread.is_alive():
        capturing = True
        thread = threading.Thread(target=start_capture)
        thread.start()

def stop_capture():
    global capturing
    capturing = False

def train_model():
    encodings = []
    names = []

    for person_folder in os.listdir(data_path):
        person_path = os.path.join(data_path, person_folder)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            name = person_folder

            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)[0]
            encodings.append(encoding)
            names.append(name)

    known_encodings = encodings
    known_names = names

    model_data = {'encodings': known_encodings, 'names': known_names}
    with open(model_path, 'wb') as model_file:
        pickle.dump(model_data, model_file)

    print('Modelo entrenado y guardado.')

def test_model():
    with open(model_path, 'rb') as model_file:
        model_data = pickle.load(model_file)

    known_encodings = model_data['encodings']
    known_names = model_data['names']

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Desconocido"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            color = (0, 255, 0)  # Por defecto, verde para personas conocidas

            if name == "Desconocido":
                color = (0, 0, 255)  # Rojo para personas desconocidas

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Reconocimiento Facial en Tiempo Real', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def close_app():
    global capturing
    capturing = False
    if thread and thread.is_alive():
        thread.join()  # Esperar a que el hilo termine
    cam.release()
    cv2.destroyAllWindows()
    root.destroy()

name_entry = Entry(root, textvariable=name_var, width=30)
name_entry.pack()

btn_capture_video = Button(root, text="Capturar desde Video", command=capture_video)
btn_capture_video.pack()

btn_stop_capture = Button(root, text="Detener Captura", command=stop_capture)
btn_stop_capture.pack()

btn_train = Button(root, text="Entrenar", command=train_model)
btn_train.pack()

btn_test = Button(root, text="Probar Modelo", command=test_model)
btn_test.pack()

root.protocol("WM_DELETE_WINDOW", close_app)  # Capturar evento de cierre de ventana
root.mainloop()

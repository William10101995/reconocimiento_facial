import face_recognition
import cv2
import numpy as np
from tkinter import *
from tkinter import ttk  # Importar ttk para Treeview
import os
import pickle

data_path = 'images'
model_path = 'trained_model.pkl'

root = Tk()
root.title("Reconocimiento Facial")

name_var = StringVar()


# Función para tomar fotografías

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

# Función para probar el modelo
# Función para tomar fotografías
def take_picture(name):
    person_folder = os.path.join(data_path, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

        cam = cv2.VideoCapture(0)
        img_counter = 0

        while True:
            ret, frame = cam.read()

            if img_counter == 10:
                break

            img_name = os.path.join(person_folder, f"img_{img_counter}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Imagen {img_counter} guardada para {name}!")
            img_counter += 1

        cam.release()

# Función para probar el modelo
def test_model():
    with open(model_path, 'rb') as model_file:
        model_data = pickle.load(model_file)

    known_encodings = model_data['encodings']
    known_names = model_data['names']

    def update_gui():
        ret, frame = cam.read()

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_encoding = face_encodings[i]
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Desconocido"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

                # Buscar si la persona ya existe en la Treeview
                item_id = None
                for item in result_tree.get_children():
                    if result_tree.item(item, "values")[0] == name:
                        item_id = item
                        break

                # Insertar o actualizar la entrada en la Treeview
                if item_id is None:
                    result_tree.insert("", "end", values=(name, "Conocido"))
                else:
                    result_tree.item(item_id, values=(name, "Conocido"))

                color = (0, 255, 0)  # Por defecto, verde para personas conocidas
            else:
                # Agregar a la Treeview como desconocido
                result_tree.insert("", "end", values=("Desconocido", "Desconocido"))
                color = (0, 0, 255)  # Rojo para personas desconocidas

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Reconocimiento Facial en Tiempo Real', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
        else:
            # Programar la próxima actualización después de 10 milisegundos
            root.after(10, update_gui)

    cam = cv2.VideoCapture(0)
    # Iniciar la primera actualización después de 10 milisegundos
    root.after(10, update_gui)

# Frame para la captura de imágenes
capture_frame = Frame(root, padx=10, pady=10)
capture_frame.pack()

name_label = Label(capture_frame, text="Nombre:")
name_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

name_entry = Entry(capture_frame, textvariable=name_var, width=20)
name_entry.grid(row=0, column=1, padx=5, pady=5)

def capture_images():
    name = name_var.get().strip()
    if not name:
        print("Por favor, ingrese un nombre antes de capturar imágenes.")
        return

    take_picture(name)

capture_btn = Button(capture_frame, text="Capturar Imágenes", command=capture_images)
capture_btn.grid(row=0, column=2, padx=5, pady=5)

# Frame para el entrenamiento del modelo
train_frame = Frame(root, padx=10, pady=10)
train_frame.pack()

train_label = Label(train_frame, text="Entrenamiento del Modelo:")
train_label.grid(row=0, column=0, padx=5, pady=5)

train_btn = Button(train_frame, text="Entrenar Modelo", command=train_model)
train_btn.grid(row=0, column=1, padx=5, pady=5)

# Frame para la prueba del modelo
test_frame = Frame(root, padx=10, pady=10)
test_frame.pack()

test_label = Label(test_frame, text="Prueba del Modelo:")
test_label.grid(row=0, column=0, padx=5, pady=5)

# Crear Treeview para mostrar resultados
result_tree = ttk.Treeview(test_frame, columns=("Name", "Status"), show="headings")
result_tree.heading("Name", text="Nombre")
result_tree.heading("Status", text="Estado")
result_tree.grid(row=1, column=0, padx=5, pady=5)

test_btn = Button(test_frame, text="Probar Modelo", command=test_model)
test_btn.grid(row=0, column=1, padx=5, pady=5)

root.mainloop()
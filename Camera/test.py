import cv2
import face_recognition
import numpy as np
import os
import pyautogui
import keyboard
import threading

# Função para capturar e salvar a imagem do rosto
def capture_and_save_face(face_image, save_folder_path):
    new_face_name = input("Nome da nova pessoa: ")
    save_path = os.path.join(save_folder_path, f"{new_face_name}.jpg")
    cv2.imwrite(save_path, face_image)
    print(f"Imagem salva como {save_path}")

# Carregar todas as imagens de referência da pasta e calcular a média dos encodings por pessoa
def load_all_reference_images(reference_folder_path):
    known_face_encodings = []
    known_face_names = []
    face_encodings_dict = {}

    for filename in os.listdir(reference_folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(reference_folder_path, filename)
            person_name = filename.split("_")[0]

            try:
                reference_image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(reference_image)

                if face_encodings:
                    if person_name in face_encodings_dict:
                        face_encodings_dict[person_name].append(face_encodings[0])
                    else:
                        face_encodings_dict[person_name] = [face_encodings[0]]

            except Exception as e:
                print(f"Não foi possível carregar {filename}: {e}")

    for person_name, encodings in face_encodings_dict.items():
        avg_encoding = np.mean(encodings, axis=0)
        known_face_encodings.append(avg_encoding)
        known_face_names.append(person_name)

    return known_face_encodings, known_face_names

# Compare face com as imagens de referência
def compare_faces(known_face_encodings, face_encoding_to_check):
    return face_recognition.compare_faces(known_face_encodings, face_encoding_to_check)

# Thread para reconhecimento facial
def face_recognition_thread():
    global frame_counter, known_face_encodings, known_face_names, video_capture

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Inverter a imagem horizontalmente
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        frame_counter += 1 

        if frame_counter % 10 == 0: #era 5 antes
            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_face_image)

                if face_encodings:
                    face_encoding = face_encodings[0]
                    matches = compare_faces(known_face_encodings, face_encoding)

                    if True in matches:
                        match_index = matches.index(True)
                        matched_name = known_face_names[match_index]
                        pyautogui.press('q')
                        print(f"Você é {matched_name}")
                    else:
                        continue

                break  # Para a captura após detectar a primeira face

# Configurações iniciais
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

reference_folder_path = 'C:/Camera/Camera/fotos'
save_folder_path = 'C:/Camera/Camera/fotos'
known_face_encodings, known_face_names = load_all_reference_images(reference_folder_path)

frame_counter = 0

# Iniciar a thread de reconhecimento facial
recognition_thread = threading.Thread(target=face_recognition_thread)
recognition_thread.start()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Inverter a imagem horizontalmente
    frame = cv2.flip(frame, 1)

    if keyboard.is_pressed('u'):
        print("Pessoa desconhecida, salvando nova imagem...")
        capture_and_save_face(frame, reference_folder_path)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
video_capture.release()
cv2.destroyAllWindows()

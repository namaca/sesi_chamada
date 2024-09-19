import cv2
import face_recognition
import numpy as np
import os
import pyautogui
import keyboard

# Função para capturar e salvar a imagem do rosto
def capture_and_save_face(face_image, save_folder_path):
    # Solicitar nome para salvar a nova face
    new_face_name = input("Nome da nova pessoa: ")
    
    # Criar o caminho completo para salvar a imagem
    save_path = os.path.join(save_folder_path, f"{new_face_name}.jpg")
    
    # Salvar a imagem capturada
    cv2.imwrite(save_path, face_image)
    print(f"Imagem salva como {save_path}")

# Carregar todas as imagens de referência da pasta e calcular a média dos encodings por pessoa
def load_all_reference_images(reference_folder_path):
    known_face_encodings = []
    known_face_names = []
    
    # Dicionário para armazenar encodings por pessoa
    face_encodings_dict = {}
    
    for filename in os.listdir(reference_folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(reference_folder_path, filename)
            person_name = filename.split("_")[0]  # Assume que o nome da pessoa está no nome do arquivo
            
            try:
                reference_image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(reference_image)
                
                if face_encodings:
                    # Se o nome da pessoa já está no dicionário, adicione o encoding
                    if person_name in face_encodings_dict:
                        face_encodings_dict[person_name].append(face_encodings[0])
                    else:
                        face_encodings_dict[person_name] = [face_encodings[0]]
                    
            except Exception as e:
                print(f"Não foi possível carregar {filename}: {e}")
    
    # Calcular a média dos encodings para cada pessoa
    for person_name, encodings in face_encodings_dict.items():
        avg_encoding = np.mean(encodings, axis=0)  # Calcula a média dos encodings
        known_face_encodings.append(avg_encoding)
        known_face_names.append(person_name)
    
    return known_face_encodings, known_face_names

# Compare face com as imagens de referência
def compare_faces(known_face_encodings, face_encoding_to_check):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check)
    return matches

# Carregar o classificador Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Captura de vídeo
video_capture = cv2.VideoCapture(0)  # Use 0 para a câmera padrão ou insira o caminho para um arquivo de vídeo

# Caminho da pasta com imagens de referência
reference_folder_path = 'C:/Camera/Camera/fotos'  # Substitua pelo caminho da pasta de imagens de referência
save_folder_path = 'C:/Camera/Camera/fotos'  # Substitua pelo caminho onde as novas imagens serão salvas

# Carregar os encodings das imagens de referência e seus nomes
known_face_encodings, known_face_names = load_all_reference_images(reference_folder_path)

frame_counter = 0  # Contador de frames

while True:
    # Captura frame a frame
    if keyboard.is_pressed('u'):  # Verifica se a tecla 'q' foi pressionada
        print("Pessoa desconhecida, salvando nova imagem...")
        capture_and_save_face(face_image, reference_folder_path);
        
    ret, frame = video_capture.read()
    
    if not ret:
        break

    # Inverter a imagem horizontalmente para corrigir o efeito espelho
    frame = cv2.flip(frame, 1)
    
    # Conversão para escala de cinza para melhorar a detecção
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecção de faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    cv2.putText(frame, 'Procurando pessoa...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    frame_counter += 1  # Incrementa o contador de frames

    # Processar a cada 5 frames
    if frame_counter % 5 == 0:
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face_image)
            
            if face_encodings:
                face_encoding = face_encodings[0]
                # Comparar a face detectada com todas as imagens de referência
                matches = compare_faces(known_face_encodings, face_encoding)
                
                if True in matches:
                    match_index = matches.index(True)  # Índice do primeiro match encontrado
                    matched_name = known_face_names[match_index]  # Nome da pessoa correspondente
                    
                    pyautogui.press('q')  # Pressiona a tecla 'q'
                    print(f"Você é {matched_name}")
                else:
                    continue
                
                # Salvar a imagem capturada
                break  # Para a captura após detectar a primeira face

    # Exibir o frame com as deteções
    cv2.imshow('Video', frame)
    
    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
video_capture.release()
cv2.destroyAllWindows()

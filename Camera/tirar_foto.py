import cv2
import os
import time
import keyboard  # Biblioteca para detectar pressionamento de teclas

# Função para salvar a imagem do rosto com um nome incrementado
def save_face_image(face_image, save_folder_path, person_name, image_id):
    # Criar o nome do arquivo com o ID
    save_path = os.path.join(save_folder_path, f"{person_name}_{image_id}.jpg")
    
    # Salvar a imagem capturada
    cv2.imwrite(save_path, face_image)
    print(f"Imagem salva como {save_path}")

# Solicitar o nome do usuário para salvar as fotos
person_name = input("Nome da pessoa: ")

# Caminho da pasta onde as fotos serão salvas
save_folder_path = 'C:/Camera/Camera/fotos'  # Substitua pelo caminho desejado

# Verifique se a pasta existe, se não, crie-a
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# Carregar o classificador Haar Cascade para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar captura de vídeo
video_capture = cv2.VideoCapture(0)  # Use 0 para a câmera padrão

# ID da imagem
image_id = 1

# Definir o número máximo de fotos a serem tiradas
max_images = 5

# Definir o delay entre as capturas (em segundos)
capture_delay = 0.5 # Exemplo: 1.5 segundos de delay entre as fotos

# Variável para controlar quando começar a tirar fotos
taking_photos = False

while True:
    # Capturar frame por frame
    ret, frame = video_capture.read()
    
    if not ret:
        break

    # Espelhar a imagem horizontalmente
    frame = cv2.flip(frame, 1)

    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos no frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Verifica se a tecla "u" foi pressionada
    if keyboard.is_pressed('u'):
        taking_photos = True
        print("Iniciando a captura de fotos...")

    # Se houver rostos detectados e estiver no modo de captura de fotos
    if taking_photos and len(faces) > 0 and image_id <= max_images:
        # Para cada rosto detectado
        for (x, y, w, h) in faces:
            # Desenhar um retângulo ao redor do rosto detectado (opcional)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extrair a imagem do rosto
            face_image = frame[y:y+h, x:x+w]
            
            # Salvar a imagem do rosto
            save_face_image(face_image, save_folder_path, person_name, image_id)
            
            # Incrementar o ID da imagem para a próxima captura
            image_id += 1
            
            # Adicionar um pequeno delay entre as capturas
            time.sleep(capture_delay)
            
            # Parar de tirar fotos se já tiver capturado o número máximo
            if image_id > max_images:
                print(f"Captura de {max_images} imagens concluída.")
                taking_photos = False  # Parar o processo de captura
                break

    # Exibir o frame com as detecções (espelhado)
    cv2.imshow('Webcam - Espelhada', frame)

    # Pressione 'q' para sair manualmente, mesmo antes de tirar todas as fotos
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
video_capture.release()
cv2.destroyAllWindows()

import cv2
import os

# Diretório para salvar as imagens capturadas
output_dir = 'captured_images'

# Verifica se o diretório de saída existe, caso contrário, cria
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Inicializa o contador de imagens
image_count = 0

# Inicializa a câmera
cap = cv2.VideoCapture(0)

while True:
    # Captura um frame da câmera
    ret, frame = cap.read()

    # Redimensiona o frame para 400x400 pixels
    frame = cv2.resize(frame, (400, 400))

    # Converte o frame para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Exibe o frame
    cv2.imshow('Capture Images for Registration', frame)

    # Aguarda a tecla 's' para salvar a imagem capturada
    key = cv2.waitKey(1)
    if key == ord('s'):
        # Salva a imagem capturada no diretório de saída
        image_path = os.path.join(output_dir, f'image_{image_count}.jpg')
        cv2.imwrite(image_path, gray_frame)
        print(f'Image saved: {image_path}')
        image_count += 1
    elif key == 27:  # Esc key
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()

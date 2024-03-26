import cv2
import numpy as np

# Carregar o modelo treinado
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.xml')  # Caminho para o arquivo XML do modelo treinado

# Inicializar o classificador de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a câmera
cap = cv2.VideoCapture(0)  # Use 0 para a câmera padrão, ou especifique o número da câmera

while True:
    # Capturar frame da câmera
    ret, frame = cap.read()

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Para cada rosto detectado, realizar reconhecimento facial
    for (x, y, w, h) in faces:
        # Extrair a região de interesse (ROI) do rosto
        roi_gray = gray[y:y+h, x:x+w]

        # Realizar o reconhecimento facial na ROI
        id_, confidence = recognizer.predict(roi_gray)

        # Verificar se a confiança do reconhecimento é alta o suficiente
        if confidence < 70:  # Ajuste este valor conforme necessário
            # Mostrar o ID do funcionário reconhecido na imagem
            cv2.putText(frame, str(id_), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Caso não seja possível identificar o funcionário, exibir "Desconhecido"
            cv2.putText(frame, "Desconhecido", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Desenhar um retângulo ao redor do rosto detectado
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Exibir o frame com os rostos detectados e IDs reconhecidos
    cv2.imshow('Reconhecimento Facial', frame)

    # Aguardar pela tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar a janela
cap.release()
cv2.destroyAllWindows()

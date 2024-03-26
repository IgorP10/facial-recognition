import cv2
import os
import numpy as np
import csv

# Pegar Diretório contendo as imagens dos funcionários
script_dir = os.path.dirname(os.path.abspath(__file__)) # Pega o diretório atual do script
dataset_dir = os.path.join(script_dir, '..', 'captured_images') # Pega o diretório das imagens

# Lista para armazenar caminhos das imagens e seus rótulos (IDs dos funcionários)
image_paths = []
labels = []

# Caminho para o arquivo CSV
csv_file_path = os.path.join(dataset_dir, 'faces.csv')

# Lê os dados do arquivo CSV
with open(csv_file_path, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_paths.append(row['image_path'])
        labels.append(int(row['label']))

# Inicializa os vetores de características e labels
faces = []
training_labels = []  # Renomeando para evitar sobreposição de variáveis

# Prepara o reconhecimento facial LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Função para carregar as imagens e extrair características
def prepare_training_data():
    for image_path, label in zip(image_paths, labels):
        # Caminho completo da imagem
        image_path_full = os.path.join(dataset_dir, image_path)
        
        # Carrega a imagem em escala de cinza
        image = cv2.imread(image_path_full, cv2.IMREAD_GRAYSCALE)

        # Adiciona a imagem e o label aos vetores
        faces.append(image)
        training_labels.append(label)  # Alterado o nome da variável aqui

# Prepara os dados de treinamento
prepare_training_data()

# Converte os vetores de características e labels para arrays numpy
faces = np.array(faces, dtype='object')
training_labels = np.array(training_labels)  # Alterado o nome da variável aqui

faces = faces.astype(np.uint8)
training_labels = training_labels.astype(np.int32)

# Treina o modelo de reconhecimento facial
recognizer.train(faces, training_labels)  # Alterado o nome da variável aqui

# Salva o modelo treinado em um arquivo XML
recognizer.save('trained_model.xml')

print('Training completed successfully!')

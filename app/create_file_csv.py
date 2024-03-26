import csv
import os

# Pegar Diretório contendo as imagens dos funcionários
script_dir = os.path.dirname(os.path.abspath(__file__)) # Pega o diretório atual do script
dataset_dir = os.path.join(script_dir, '..', 'captured_images') # Pega o diretório das imagens

# Lista de imagens e seus rótulos (IDs dos funcionários)
image_label_list = [
    ('image_0.jpg', 55),
    ('image_9.jpg', 100)
]

# Caminho para o arquivo CSV
csv_file_path = os.path.join(dataset_dir, 'faces.csv')

# Escreve os dados no arquivo CSV
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['image_path', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for image_path, label in image_label_list:
        writer.writerow({'image_path': image_path, 'label': label})

print('CSV file created successfully!')

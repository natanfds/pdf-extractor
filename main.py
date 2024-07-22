from pdf2image import convert_from_path
import tempfile
import os
import cv2
from uuid import uuid4
import pytesseract
from sklearn.cluster import DBSCAN
import numpy as np
from itertools import chain
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import sys

path_to_pdf=""
# Verifica se algum argumento foi passado
if len(sys.argv) > 1:
    path_to_pdf = sys.argv[1]
    print(f"File: {path_to_pdf}")
else:
    print("No file found")
    exit(1)

if(not os.path.exists(path_to_pdf)):
    print(f"File {path_to_pdf} not found")
    exit(1)

load_dotenv()
DEBUG_MODE = os.getenv('DEBUG', 0) == '1'

# Definir o caminho do arquivo PDF
pdf_path = './arquivo.pdf'

# Converter PDF em uma lista de imagens
images = convert_from_path(pdf_path)

saved_images = []

for i, image in enumerate(images):
  with tempfile.TemporaryDirectory() as tmp:
    img_id = uuid4()
    image_path = os.path.join(tmp, f'imagem_{i}_{img_id}.png') if not DEBUG_MODE else f'output/page_{i}_{img_id}.png'
    print(f'Salvando imagem {image_path}')
    image.save(image_path, 'PNG')
    saved_images.append(image_path)
    
    image = cv2.imread(image_path)
    
    #Transformar a imagem em máscara preto e branco
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw_map = cv2.Laplacian(gray_image, cv2.CV_64F)
    bw_map = cv2.convertScaleAbs(bw_map)
    bw_map = cv2.GaussianBlur(bw_map, (13,13), 10)
    bw_map = cv2.bitwise_not(bw_map)
    bw_map = cv2.inRange(bw_map, 20, 250)
    if(DEBUG_MODE):
        cv2.imwrite(f'output/page_{i}_bw_{img_id}.png', bw_map)
    # Encontrar contornos na imagem
    contours, _ = cv2.findContours(bw_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    def is_rect_valid(rect): 
        _, _ , w, h = rect
        return  w > 50 and h > 10
    rects = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 100]
    rects = [rect for rect in rects if is_rect_valid(rect)]

    counter = 0
    output_data = []
    
    rect_coords = [[x,y, x+w, y+h] for x, y, w, h in rects]
    image_pil = Image.open(image_path)
    image_draw = ImageDraw.Draw(image_pil, 'RGBA')
    for coords in rect_coords:

        image_draw.rectangle(coords, outline='red')
    
    image_pil.save(f'output/page_{i}_debug_rects_{img_id}.png')
    
    # Extrair texto de cada retângulo
    for x, y, w, h in rects:
        rect = image[y:y+h, x:x+w]
        counter += 1
        extracted_text = pytesseract.image_to_string(rect)
        print(f'Encontrado texto válido: {extracted_text}')
        coords = (x, y, w, h)
        print(f'Coordenadas do retângulo: {coords}')
        
        output_data.append({
          'text': extracted_text,
          'coords': coords
        })
    
    if(len(output_data) == 0):
      continue
    
    # ordenar colunas horizontal e verticalmente
    all_x_values = sorted([data['coords'][0] for data in output_data])
    all_x_values = np.array(all_x_values).reshape(-1, 1)
    dbscan = DBSCAN(eps=5, min_samples=2).fit(all_x_values)
    
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'Número de colunas: {n_clusters}')
    

    clusters = {}
    for label, x_value in zip(labels, all_x_values.flatten()):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(x_value)

    # Calcular máximo e mínimo para cada cluster
    group_limits = []
    for label, values in clusters.items():
        if label == -1:  # Ignorar ruído
            continue
        max_value = max(values)
        min_value = min(values)
        group_limits.append((label, min_value, max_value))
        print(f'Cluster {label}: Máximo = {max_value}, Mínimo = {min_value}')
    
    group_limits = sorted(group_limits, key=lambda x: x[1])
    
    # formando grupos
    pre_sorted_data = []
    for limit in group_limits:
        label, min_value, max_value = limit
        pre_sorted_data.append([data for data in output_data if min_value -200 <= data['coords'][0] <= max_value + 200])

    pre_sorted_data = [ sorted(data, key=lambda x: x['coords'][1]) for data in pre_sorted_data ]
    pre_sorted_data = list(chain.from_iterable(pre_sorted_data))
    
    data = [e['text'] for e in pre_sorted_data]
    data = '\n\n-----------------\n\n'.join(data)
    with open(f'output/page_{i}.txt', 'w') as f:
        f.write(data)
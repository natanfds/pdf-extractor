from pdf2image import convert_from_path
import tempfile
import os
import cv2
from uuid import uuid4
import pytesseract
import re
from sklearn.cluster import DBSCAN
import numpy as np
from itertools import chain

# Função para verificar se um retângulo está contido em outro
def is_contained(ret1, ret2):
    x1, y1, w1, h1 = ret1
    x2, y2, w2, h2 = ret2
    return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2
  

# Definir o caminho do arquivo PDF
pdf_path = './arquivo.pdf'

# Converter PDF em uma lista de imagens
images = convert_from_path(pdf_path)

saved_images = []

for i, image in enumerate(images):
  with tempfile.TemporaryDirectory() as tmp:
    img_id = uuid4()
    image_path = os.path.join(tmp, f'imagem_{i}_{img_id}.png')
    print(f'Salvando imagem {image_path}')
    image.save(image_path, 'PNG')
    saved_images.append(image_path)
    
    image = cv2.imread(image_path)
    
    #Transformar a imagem em máscara preto e branco
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw_map = cv2.Laplacian(gray_image, cv2.CV_64F)
    bw_map = cv2.convertScaleAbs(bw_map)
    bw_map = cv2.GaussianBlur(bw_map, (15,15), 10)
    bw_map = cv2.bitwise_not(bw_map)
    bw_map = cv2.inRange(bw_map, 20, 250)

    # Encontrar contornos na imagem
    contornos, _ = cv2.findContours(bw_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Coletar todos os retângulos
    rects = [cv2.boundingRect(contorno) for contorno in contornos if cv2.contourArea(contorno) > 100]

    # Remover retângulos completamente contidos em outros
    filtered_rects = [r for i, r in enumerate(rects) if not any(is_contained(r, r2) for j, r2 in enumerate(rects) if i != j)]

    counter = 0
    output_data = []
    for x, y, w, h in filtered_rects:
        rect = image[y:y+h, x:x+w]
        file_name = f'out/rect_{img_id}_{counter}.png'
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
    
    # ordenar colunas
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
    with open(f'out/page_{i}.txt', 'w') as f:
        f.write(data)
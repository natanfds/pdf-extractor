from pdf2image import convert_from_path
import tempfile
import os
import cv2
from uuid import uuid4

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
    bw_map = cv2.GaussianBlur(bw_map, (21,21), 10)
    bw_map = cv2.bitwise_not(bw_map)
    bw_map = cv2.inRange(bw_map, 20, 250)

    # Encontrar contornos na imagem
    contornos, _ = cv2.findContours(bw_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Coletar todos os retângulos
    rects = [cv2.boundingRect(contorno) for contorno in contornos if cv2.contourArea(contorno) > 100]

    # Remover retângulos completamente contidos em outros
    filtered_rects = [r for i, r in enumerate(rects) if not any(is_contained(r, r2) for j, r2 in enumerate(retangulos) if i != j)]

    counter = 0
    for x, y, w, h in filtered_rects:
        rect = image[y:y+h, x:x+w]
        file_name = f'out/rect_{img_id}_{counter}.png'
        counter += 1
        
        if w > 100 and h > 50:
          cv2.imwrite(file_name, rect)
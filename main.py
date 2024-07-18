from pdf2image import convert_from_path
import tempfile
import os

# Definir o caminho do arquivo PDF
pdf_path = './arquivo.pdf'

# Converter PDF em uma lista de imagens
images = convert_from_path(pdf_path)

saved_images = []
# Iterar sobre as imagens e salvá-las
for i, image in enumerate(images):
  with tempfile.TemporaryDirectory() as tmp:
    image_path = os.path.join(tmp, f'imagem_{i}.png')
    image.save(image_path, 'PNG')
    print(f'Imagem {i} salva em {image_path}')
    saved_images.append(image_path)


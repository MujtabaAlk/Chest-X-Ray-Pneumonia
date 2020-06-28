from PIL import Image

image = Image.open('images/test.jpg')
image_final = image.resize((32, 32), Image.BILINEAR)
image_final.save('images/out.jpg')

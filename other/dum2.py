import PIL
from dataset import Images
image = PIL.Image.open("10815824_2997e03d76.jpg")

width, height = image.size

print(width, height)
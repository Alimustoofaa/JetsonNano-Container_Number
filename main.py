import cv2
from src.process import main_process

image = cv2.imread('img/1.jpg')
results = main_process(image)
print(results)
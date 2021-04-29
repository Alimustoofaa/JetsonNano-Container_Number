'''
@author     : Ali Mustofa HALOTEC
@module     : Container Number Detection YoloV5
@Created on : 29 April 2021
'''

import os
import torch
import requests
from config import *
from tqdm import tqdm
from pathlib import Path

class ContainerNumberPrediction:
    '''
    Load custom model Yolo v5
    in directory model/model_container_number.pt
    '''
    def __init__(self):
        self.model_path = os.path.join(DIRECTORY_MODEL, detection_model['filename'])
        self.check_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=self.model_path)
        self.model.to(self.device)

    def check_model(self):
        '''
        Checking model in model_path
        download model if file not found
        '''
        Path(DIRECTORY_MODEL).mkdir(parents=True, exist_ok=True)
        if not os.path.isfile(self.model_path):
            print('Downloading container number detection model, please wait.')
            response = requests.get(detection_model['url'], stream=True)
            progress = tqdm(response.iter_content(1024), 
                        f'Downloading {detection_model["filename"]}', 
                        total=detection_model['file_size'], unit='B', 
                        unit_scale=True, unit_divisor=1024)
            with open(self.model_path, 'wb') as f:
                for data in progress:
                    f.write(data)
                    progress.update(len(data))
                print('Done downloaded container number detection model.')
        else:
            print('Load container number detection model.')

    def filter_and_crop(self, img, results, min_confidence=0.0):
        '''
        Format result([tensor([[151.13147, 407.76913, 245.91382, 454.27802,   0.89075,   0.00000]])])
        Filter min confidence prediction and classes id/name
        Cropped image and get index max value confidence lavel
        and retrun image, confidence
        '''
        confidence_list = list()
        image_crop = list()
        results_format = results.xyxy
        for i in range(len(results_format)):
            classes_name = classes[int(results_format[0][i][-1])]
            confidence = float(results_format[0][i][-2])
            if confidence > min_confidence and classes_name == 'container_number':
                x1, y1 = int(results_format[0][i][0]), int(results_format[0][i][1])
                x2, y2 = int(results_format[0][i][2]), int(results_format[0][i][3])
                cropped_img = img[y1-10:y2+10, x1-10:x2+10]
                image_crop.append(cropped_img)
                confidence_list.append(confidence)
                
        confidence_max_index = max(range(len(confidence_list)), key=confidence_list.__getitem__)
        image_crop_max = image_crop[confidence_max_index]
        return image_crop_max, confidence_list[confidence_max_index]

    def prediction(self, image):
        '''
        Prediction image object detectionn YoloV5
        output prediction label (xmin,  ymin, xmax, ymax, confidence, class, name)
        '''
        results = self.model(image)
        return results

# model_container_number = ContainerNumberPrediction()
# image = cv2.imread('img/1.jpg')
# results = model_container_number.prediction(image)
# new_img, confidence = model_container_number.filter_and_crop(img=image, results=results, min_confidence=0.3)
# print(confidence)
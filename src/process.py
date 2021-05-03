import cv2

from .schema.config_ocr import ConfigOcr
from .app.container_number_detection import ContainerNumberPrediction
from .app.container_number_ocr import OpticalCharacterRecognition

container_number_detection = ContainerNumberPrediction()
container_number_ocr = OpticalCharacterRecognition()

def detection(image):
	result_detection = container_number_detection.prediction(image)
	new_img, conf = container_number_detection.filter_and_crop(
		img=image, results=result_detection, min_confidence=0.3
	)
	print(f'Got container number detection confidence : {conf}')
	return new_img

def detect_char(image):
	return container_number_ocr.detect_char(image)

def read_text(image):
	config = ConfigOcr(
		beam_width      = 20,
		batch_size      = 10,
		text_threshold  = 0.4,
		link_threshold  = 0.7,
		low_text        = 0.4,
		slope_ths       = 0.9
	)
	return container_number_ocr.ocr_image(image, config)

def resize(image):
	scale_percent = 180
	width = int(image.shape[1] * scale_percent / 100)
	height = int(image.shape[0] * scale_percent / 100)
	dim = (width, height)
	return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def main_process(image):
	image_detection = detection(image)
	# resize image
	image_crop = resize(image_detection) if image_detection.shape[1] <= 175 else image
	# detect character
	detected_char = detect_char(image_crop)
	image_crop = image_crop if detected_char else image
	print(detected_char)

import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3
from keras.models import load_model
from texttospeech.speechengine import SpeechEngine


curdir = os.getcwd()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
prediction = None
model_file = os.path.join(os.getcwd(), "/cnn_model_keras2.h5")
print(model_file)
model = load_model('/home/ashis/gesture2speech/src/gesturetotext/cnn_model_keras2.h5')

def get_image_size():
	img = cv2.imread('/home/ashis/gesture2speech/src/gesturetotext/gestures/0/100.jpg',0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("/home/ashis/gesture2speech/src/gesturetotext/gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def split_sentence(text, num_of_words):
	'''
	Splits a text into group of num_of_words
	'''
	list_words = text.split(" ")
	length = len(list_words)
	splitted_sentence = []
	b_index = 0
	e_index = num_of_words
	while length > 0:
		part = ""
		for word in list_words[b_index:e_index]:
			part = part + " " + word
		splitted_sentence.append(part)
		b_index += num_of_words
		e_index += num_of_words
		length -= num_of_words
	return splitted_sentence

def put_splitted_text_in_blackboard(blackboard, splitted_text):
	y = 200
	for text in splitted_text:
		cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		y += 50

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def recognize():
	global prediction
	cam = cv2.VideoCapture(1)
	if cam.read()[0] == False:
		cam = cv2.VideoCapture(0)
	hist = get_hand_hist()
	x, y, w, h = 500, 100, 300, 300
	while True:
		text = ""
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		imgCrop = img[y:y+h, x:x+w]
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst,-1,disc,dst)

		# kernel1 = np.ones ((8,8),np.uint8)
		# erosion = cv2.erode(dst, kernel1, iterations =1)
		# kernel2 = np.ones((12,12),np.uint8)
		# dst = cv2.dilate(erosion ,kernel2,iterations = 1)

		blur = cv2.GaussianBlur(dst, (11,11), 0)
		blur = cv2.medianBlur(blur, 15)
		thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		thresh = cv2.merge((thresh,thresh,thresh))
		thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
		thresh = thresh[y:y+h, x:x+w]
		contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
		# print(contours)
		# print(type(contours))
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				save_img = thresh[y1:y1+h1, x1:x1+w1]
				
				if w1 > h1:
					save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
			
				pred_probab, pred_class = keras_predict(model, save_img)
				print(pred_class, pred_probab)
				
				if pred_probab*100 > 80:
					text = get_pred_text_from_db(pred_class)
					if text:
						speek = SpeechEngine()
						speek.speech(text)
					print(text)
		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		splitted_text = split_sentence(text, 2)
		put_splitted_text_in_blackboard(blackboard, splitted_text)
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
		res = np.hstack((img, blackboard))
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)
		keypress = cv2.waitKey(1) & 0xFF
		if keypress == ord('q'):
			break

keras_predict(model, np.zeros((50, 50), dtype=np.uint8))		


if __name__ == "__main__":
	recognize()
# import the necessary packages
from imutils.contours import sort_contours
import numpy as np
import pandas as pd
from  keras.preprocessing import image
from keras.models import model_from_json
import keras
import tensorflow as tf
import pathlib
import cv2
import imutils
import os
labels=['A','B','C']
# construct the argument parser and parse the arguments
json_file=open("model.json",'r')
loaded_model_json=json_file.read()
json_file.close()
model=model_from_json(loaded_model_json)
model.load_weights("model.h5")
model.compile(optimizer='Adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

image_name=input('Enter Image Name: ')
# load the input image from disk, convert it to grayscale, and blur
# it to reduce noise
image = cv2.imread(image_name)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []
chr=''
for c in cnts:
# compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    # filter out bounding boxes, ensuring they are neither too small
    print(x,' ',y,' ',w,' ',h)
    new=image[y:y+h,x:x+w]
    cv2.imwrite('temp.png',new)
    img = cv2.imread('temp.png', cv2.IMREAD_UNCHANGED)
    new=cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
    cv2.imwrite('temp.png',new)
    test_image=keras.preprocessing.image.load_img('temp.png',target_size=(64,64))
    test_image=keras.preprocessing.image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    prediction=model.predict(test_image)
    score = tf.nn.softmax(prediction[0])
    chr=chr+labels[np.argmax(score)]

print(chr)
os.remove('temp.png')

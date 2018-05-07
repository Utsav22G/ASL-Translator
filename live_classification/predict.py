import cv2
from cv2 import imread
from keras.models import load_model
import numpy as np

model = load_model('weights/model.h5')


img = imread('Data/TestData/X_11.jpg') #try converting into a numpy array
img = img.astype(np.float32)/255.0
img = img[:,:,::-1]
result = model.predict(np.expand_dims(img, axis=0))
print(result)

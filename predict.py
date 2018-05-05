from cv2 import imread
from keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('model.h5')


img = Image.open('Data/OutputData/scaledoutput.jpg') #try converting into a numpy array
img = img.resize((200,200), resample=0)
img = img.save('Data/OutputData/resultingresizedimage1.jpg')
img = imread('Data/OutputData/resultingresizedimage1.jpg')
img = img.astype(np.float32)/255.0
img = img[:,:,::-1]
result = model.predict(np.expand_dims(img, axis=0))
print(result)

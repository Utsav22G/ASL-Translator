from cv2 import imread
from keras.models import load_model
from PIL import Image
import numpy as np
from gtts import gTTS
import vlc

"""
Predict
This script loads a pre-trained CNN model and classifies American Sign Language
finger spelling from a single image
Signum: Software Design SP18 Final Project
Isaac Vandor, Utsav Gupta, Diego Berny
"""

def load(trained_model):
    """ Loads a pre-trained model. """

    model = load_model(trained_model)
    return model

def predict(trained_model, test_image):
    """ Loads an image, resizes it to the size model was trained on,
    corrects the color channels to be similar to the model's channels
    and predicts the ASL alphabet. """

    img = Image.open(test_image)
    img = img.resize((200,200), resample=0)     # resize to 200x200 px
    img = img.save('Data/OutputData/temp.jpg')
    img = imread('Data/OutputData/temp.jpg')
    img = img.astype(np.float32)/255.0      # convert to float32
    img = img[:,:,::-1]         # convert from RGB to BGR

    result = trained_model.predict(np.expand_dims(img, axis=0))[0]
    return result

def find_alphabet(letter_list, letter_dict):
    """ Finds the biggest element in the list and looks for the corresponding
    key in the dictionary.

    result: list who's biggest element we're trying to find
    letter_list: dictionary whose key corresponds to the largest element """

    idx = letter_list.argmax(axis=0)    # find the index of the biggest argument

    # look for the key corresponding to the biggest argument
    decoded = [key for key, value in letter_dict.items() if value == idx]
    return decoded[0]


def say(speech, filename):
    """Initializes gTTS and saves the input text as an mp3 file."""
    tts = gTTS(text=speech, lang='en')
    tts.save(filename)

def play(speech, filename):
    """Plays the mp3 file containing the inout text."""
    say(speech, filename)
    p = vlc.MediaPlayer(filename)
    p.play()
    while True:
        pass

if __name__ == "__main__":

    model = load(trained_model='models/model.h5')
    result = predict(trained_model=model, test_image='Data/OutputData/scaledoutput.jpg')

    alphabets = {"A": 0, "B":1, "C": 2, "D":3, "E": 4, "F": 5,
                "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11,
                "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "S": 17,
                "T": 18, "U": 19, "V": 20, "W": 21, "X": 22, "Y": 23}

    alphabet = find_alphabet(letter_list=result, letter_dict=alphabets)
    print("The alphabet is: " + alphabet)
    play(alphabet, "predictfile.mp3")

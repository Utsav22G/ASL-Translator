import cv2
import gTTS
import numpy
import pandas
import sklearn


class Recognition:
    def __init__(self, frame):
        pass

    def apply_transform(self, frame):
        """
        Apply filters and transormations to extract ASL hand symbols.
        Returns symbol"""
        pass

class Classify:
    def __init__(symbol, dataset):
        pass

    def classify(self, symbol, dataset):
        """
        Uses neural net to classify ASL symbols according to the training dataset.
        Returns text
        """
        pass

class Speech:
    def __init__(text):
        pass

    def text_to_speech(self):
        """
        Use gTTS to output audio of text
        """
        pass

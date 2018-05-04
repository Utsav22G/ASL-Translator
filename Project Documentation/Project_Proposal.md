# Project Proposal

## Big Idea
We will be implementing a machine learning algorithm that recognizes ASL hand gestures and 
translates them to text which will be spoken out loud by the computer. 

__MVP:__ Translating images with ASL gestures into text and then into speech.

__Stretch goal:__ Translate a video with ASL gestures into text, and then speech.

__Super-stretch goal:__ Real-time ASL translation into text and speech.

## Learning Goals
__Diego:__ I want to learn more about how machine learning works in the context of giving it a 
large set of data and having it recognize similar patterns with computer vision.

__Utsav:__ I want to learn about the image recognition algorithms to recognize the ASL gestures as 
well as the machine learning algorithms that we’ll use to classify our data.

__Isaac:__ I am interested in applying machine learning to a complex problem that leads to a product 
that is actually beneficial to society in some way. This is a challenging project, but I can learn a
lot about how machine learning works with human input from it.

## Implementation Plan
Libraries to be used:

* [scikit-learn](http://scikit-learn.org/stable/documentation.html), [tensorflow](https://www.tensorflow.org/programmers_guide/) 
and maybe [keras](https://keras.io/) for machine learning
* [OpenCV](https://docs.opencv.org/2.4/) for ASL symbol recognition
* [gTTS](https://pypi.python.org/pypi/gTTS) and [pyttsx3 2.6](https://pypi.python.org/pypi/pyttsx3/2.6) for converting text to speech

## Project Schedule
__Week 1:__ Finding a dataset of ASL symbols, gathering data about machine learning algorithms to classify
our data, using gTTS and pyttsx3 2.6 to convert text to speech

__Week 2:__ Exploring scikit-learn, learning about basic ML algorithms and using them in scikit-learn, 
write a pattern recognition algorithm using OpenCV to recognize graphic symbols in images

__Week 3:__ Learn about intermediate-level ML algorithms and use them in scikit-learn, start exploring 
tensorflow to implement the basic algorithms we learned in Week 2, improve the symbol recognition algorithm

__Week 4:__ Write an algorithm to classify the set of ASL symbols using tensorflow, Complete the MVP

__Week 5:__ Try to recognize symbols in a video, achieve stretch goal

__Week 6:__ Optimize the algorithm to achieve super-stretch goal

## Collaboration Plan
We are planning on working in a divide-and-conquer style, where we all do parts of the project separately 
and put them together when we meet every so often. We are going to use an agile development strategy, so 
we can change what needs to be done depending on how things are going and what problems we encounter as 
quickly as possible. The meeting will be held every other day for the first two weeks and then every third 
day after that, apart from class timings.

## Risks Involved
We will be exploring lots of topics in machine learning, and we will need a large set of data for the program
to “learn” what sign language looks like properly. It is very important that it is able to get a good foundation
of data so it recognizes an new input of sign language as something it has seen before.

## Additional Course Content
We’ll be using the Image Recognition toolbox as the starting point for ASL sign recognition. The machine learning 
toolbox will be useful for intro to scikit-learn.

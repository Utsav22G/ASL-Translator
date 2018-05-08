# ASL-Translator
Software Design Final Project 4 (Spring 2018) code and documentation.

## Overview
Signum is a near real-time American Sign Language (ASL) translation tool that uses computer vision to recognize and track a user's gestures and then uses a learned model to identify the ASL character most closely correlated to that gesture. For more information, see our [project website](https://utsav22g.github.io/ASL-Translator/) or look at our [project poster](https://github.com/Utsav22G/ASL-Translator/blob/master/Project%20Documentation/Signum_Final_Poster.pdf).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

1. Clone this repo to your local machine: `git clone https://github.com/Utsav22G/ASL-Translator.git`
2. `python recognition.py` runs just the computer vision hand gesture detection
3. `python3 live_demo.py` will run the output of the CV gesture recognition comparing against a pre-trained model.

### Prerequisites

To get the keyboard up and running, please upgrade your Linux dependecies:
```
sudo apt-get update
sudo apt-get upgrade
```

Run `pip install -r requirements.txt` in order to download all the prerequisites necessary to run the program

### Project Documentation

For a better understanding of who we are and how Signum works, see our [website](https://utsav22g.github.io/ASL-Translator/). To see the program in live-action, watch this [video](https://www.youtube.com/watch?v=yB_AGuRj0Zg). For a more visual representation of the components of Signum, see our [project poster](https://github.com/Utsav22G/ASL-Translator/blob/master/Project%20Documentation/Signum_Final_Poster.pdf).

#### Built With

* [OpenCV](https://opencv.org/) - Computer vision library
* [Keras](https://keras.io) - Machine learning library
* [gTTS](http://gtts.readthedocs.io/en/latest/) - Google text-to-speech interface
* [HTML5Up!](https://html5up.net/) - Used to generate project website
* [BU ASLRP](https://www.bu.edu/asllrp/) - Used to generate dataset of ASL images

#### Contributing

Please read [Contributing.md](Contributing.md) for details on our code of conduct, and the process for submitting pull requests to us.

#### Authors
[Isaac Vandor](http://isaacvandor.com/), [Utsav Gupta](http://github.com/Utsav22G/) and [Diego Berny](https://github.com/dberny)

#### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

#### Acknowledgments

* Inspiration for this project comes from [ASL Gloves](http://www.olin.edu/news-events/2016/asl-gloves/).
* Thank you to the incredible researchers at Boston University for their work in developing an [ASL Dataset](https://www.bu.edu/asllrp/).

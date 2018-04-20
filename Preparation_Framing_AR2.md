# Architectural Review Preparation and Framing Document

## Background and Context
* We are developing a system that can detect ASL letters using OpenCV, and a neural net to determine what letter is being shown to the camera. Once we have the letter in text format, we output the letters as speech.

* As the 6th most used language in the U.S. and a complex language with its own grammar and syntax, ASL is an interesting and challenging task for realtime translation and one that will allows us to develop a project with impact beyond just the scope of the Software Design class.


## Risk Identification and Mitigation
* __Integration__
  * _Challenge:_ The code is broken into different parts and may not come together to work cohesively very easily.
  * _Mitigation:_ We have to work as a team to make sure the code functions as a whole as well as in its individual parts, so just spending time on making sure that comes together is how we will work to achieve that.

* __Robust Code__
  * _Challenge:_ As it is, the code takes a long time to run and does not always give us the result we are looking for. In one case the version of python prevents us from running, and when we can run it it doesn't save the learning it did.
  * _Mitigation:_ We have to continue debugging our code to find out why it doesn't work. It should just take extra man-hours of looking at the errors to fix what isn't working.

* __User-friendly Interface__
  * _Challenge:_ We want to make sure our website is easy to use for anyone that wants to try it out. Without that, it takes away from the reason for doing the project at all.
  * _Mitigation:_ We will ask for feedback from other Oliners, professors, and NINJAs to see what they think of how our website is organized and how it looks. Getting this kind of information while building the site will help us make sure we get it to where we want it to be without having to redo the whole thing at the end.

## Key Questions

* Signum as a website?
	* User navigates to translate page
	* Pre-trained model running in the backend
	* Video frame displays the gesture recognition and output
* What other solutions might work better from a user experience/interaction perspective?
* Would you still use Signum if it were to take a screenshot of a certain gesture and use it for comparison?


## Agenda for Technical Review Session

* First ten-fifteen minutes spent going over information presented in the slides, explaining the main parts of what we have done and what we are still working on.
  * Allow for audience to speak if they have questions/concerns about anything we are talking about from the slides

* Next five minutes have a discussion with the other teams, asking them about the questions we want to have answered

## Feedback Form
Please click [here](https://goo.gl/forms/BVY1nCyRbZVWWFBJ2) to access the feedback form.


## Link to Architecture Review Slides
Please click [here](https://docs.google.com/presentation/d/10dQnx5sBK4sAAAHAg097FJmZlo__5DoQkt3aUaCPyeA/edit?usp=sharing) to access the architecture review slides.

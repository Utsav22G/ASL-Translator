# Architectural Review Preparation and Framing Document

## Background and Context
* We are going to develop a real-time American Sign Language translation system using OpenCV and a machine learning model to understand raw ASL input and output letters and words as speech.

* As the 6th most used language in the U.S. and a complex language with its own grammar and syntax, ASL is an interesting and challenging task for realtime translation and one that will allows us to develop a project with impact beyond just the scope of the Software Design class.


## Risk Identification and Mitigation
* __Dataset__
  * _Challenge:_ We have still not found a really robust dataset of common ASL letters and words to train our program on and doing so is absolutely key to the project.
  * _Mitigation:_ If we cannot access an existing dataset by the end of this week, we will transition the program to developing Semaphore translation using openCV and machine learning.

* __Algorithm Development__
  * _Challenge:_ While we’ve begun thinking about the structure of our translation program, executing on it is our biggest technical challenge.
  * _Mitigation:_ Every member of our team will be doing the Machine Learning and Image Recognition toolboxes, which should give us the fundamentals for algorithm development in the space.

* __Time__
  * _Challenge:_ Not only do we have limited time for the development of this project, but attempting to do opencv identification, classification based on a machine learning model, and text to speech in real-time is going to be a difficult challenge requiring well-structured, concise code.
  * _Mitigation:_ We will be building and testing each block independently so that they all run as efficiently as possible. We will also be able to effectively test each module since we are building them separately and multiple pairs of eyes will be on them during our code reviews. As we’re using an agile development methodology and have a well structured timeline, we will focus on meeting those goals to avert any issues with lack of work time.

## Key Questions

* Is our project appropriately scoped?
* Is our code structured in a reasonable/effective way?
* What is going to be our biggest challenge moving forward?
* What would a reasonable MVP for this project look like?


## Agenda for Technical Review Session

* First five minutes spent going over information presented in the slides, explaining the main parts of what we are doing and how we plan to do it
  * Allow for audience to speak if they have questions/concerns about anything we are talking about from the slides
  
* Next five minutes have a discussion with the other teams, asking them about the questions we want to have answered

## Feedback Form
Please click [here](https://goo.gl/forms/i2WL2itogclpjAQ63) to access the feedback form.


## Link to Architecture Review Slides
Please click [here](https://docs.google.com/presentation/d/1L78eLBx0aYricjTrItI9goPPDkw8EYZ751lgvj6VXgg/edit?usp=sharing) to access the architecture review slides.


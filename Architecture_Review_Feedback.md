# Architecture Review Reflection

## Feedback and Decisions
_Based upon your notes from the technical review, synthesize the feedback you received addressing your key questions. How do you plan to incorporate it going forward? What new questions did you generate?_

Based upon notes from the technical review, we've heavily prioritized the recognition part of our code such that the hand detection and tracking has already been developed. A lot of feedback related to the difficulty of capturing minute differences between gestures in ASL, so we have also done research into using just the ASL alphabet or semaphores as an executable backup plan. Currently, a user-configurable thresholding system is built into the classification code such that we can precisely calibrate our hand detection and tracking each time in order to maximize the potential for success. One area that we still have questions in is the implementation of the ML aspect itself as we are not entirely sure which ML methodology would work best in this instance. Currently, our best approach for this is using a support vector machine (SVM) training method, but we do still have some questions as to whether this is the best approach. More research is needed in this area specifically

## Review Process Reflection
_How did the review go? Did you get answers to your key questions? Did you provide too much/too little context for your audience? Did you stick closely to your planned agenda, or did you discover new things during the discussion that made you change your plans? What could you do next time to have an even more effective technical review?_



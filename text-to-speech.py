import pyttsx3

# say = "The quick brown fox jumped over the lazy dog."
say = "Two roads diverged in a wood, and I, I took the one less traveled by, And that has made all the difference."

engine = pyttsx3.init()

def talk(speech):


    engine.say(speech)
    engine.runAndWait()


# voices = engine.getProperty('voices')
# for voice in voices:
#    engine.setProperty('voice', voice.id)
#    talk(say)
# engine.runAndWait()
talk(say)

from gtts import gTTS
import vlc

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

play("Hello!", "myfile.mp3")

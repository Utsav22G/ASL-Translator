from gtts import gTTS

def tts(input):
    speech = gTTS(text = input, lang = 'en', slow = False)
    f = TemporaryFile()
    speech.write_to_fp(f)
    f.close()

input = "Machine learn!"
tts(input)

#real time sentiment analysis using prior developed function (sentiment_score)
import speech_recognition as sr
from sentimentvader import sentiment_score

#creating speech recognition object
r = sr.Recognizer()
#setting minimum audio level, waiting for speech
r.energy_threshold = 4000
r.dynamic_energy_threshold = True

with sr.Microphone() as source:
    #wait 1 second, then listen to voice 
    r.adjust_for_ambient_noise(source,duration=1)
    print("Speak now...")
    audio = r.listen(source)

    try:
        #speech recognition
        text = r.recognize_google(audio,language="en-US")
        print("You said: {}".format(text))
        #sentiment analysis
        sentiment_score(text)

    except:
        #error message
        print("Could not recognize voice, try again.")




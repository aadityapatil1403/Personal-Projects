#real time speech recognition storing audio file with name
import speech_recognition as sr

#create speech recognition object
r = sr.Recognizer()

#prompt user to state name and create .wav file under their name
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    print("Please state you first and last name:")
    audio = r.listen(source)

    try:
        full_name = r.recognize_google(audio,language="en-US")
        split_name = full_name.split()
        first_name = split_name[0]
        last_name = split_name[1]
        print("Hello {}".format(full_name))

    except:
        print("Could not recognize your voice")

 #promt user to speak, recording of this saved in .wav file   
    print("Continue speaking:")
    audio2 = r.listen(source)

    try:
        text = r.recognize_google(audio2,language="en-US")
        print("You said: {}".format(text))
        with open("{}_{}.wav".format(first_name,last_name),"wb") as f:
            f.write(audio2.get_wav_data())
    except:
        print("Could not recognize your voice")

    
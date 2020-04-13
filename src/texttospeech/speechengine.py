import os
from gtts import gTTS
import subprocess
import time

class SpeechEngine():
    language = 'ne'    

    def speech(self, text):
        tts = gTTS(text, lang=self.language, slow=False)
        file_name =  time.strftime("%Y%m%d-%H%M%S") + ".mp3"
        print(file_name)
        print(type(file_name))
        tts.save(file_name)
        #print(type(audio_file))
        #audio_file = os.path.abspath(str(audio_file))
        subprocess.call(["ffplay","-nodisp","-autoexit",file_name])
        return None
	
	
        	

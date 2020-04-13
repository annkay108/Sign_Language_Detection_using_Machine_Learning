from flask import Flask, render_template, Blueprint, redirect
import os
from gesturetotext import setting_roi_hand as sett
from texttospeech.speechengine import SpeechEngine
from gesturetotext import recognize_gesture as recognizer

APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(APP_PATH, 'templates/')

app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/setting",methods=['GET','POST'])
def setting():
    # print(request)
    sett.get_hand_hist()
    return redirect("/")


@app.route("/recognize", methods=['GET','POST'])
def recognize():
    recognizer.recognize()
    return redirect('/')


@app.route('/texttospeech', methods=['GET','POST'])
def texttospeech():
    text = request.text
    speak = SpeechEngine()
    speak.speech(text)
    return redirect("/")



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8080')

"""
Sample Kivy GUI
"""

import json
import numpy as np
import cv2

import requests
from urllib.request import urlopen

from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.properties import ListProperty, StringProperty
from viewer.utils.utils import getScreenResolution


SERVER_ADR = 'http://127.0.0.1:5000'
UPDATE_RATE = 0.01


class AudioTaggerWindow(FloatLayout):
    prob_list = ListProperty([100, 100, 100, 100, 100])  # length of class probability bars
    class_list = ListProperty(['', '', '', '', ''])  # labels of class probability bars
    class_bar_height = ListProperty([0, 0, 0, 0, 0])  # simulate visibility of class probability bars
    pred_list = ListProperty([])  # selectable list of predictors
    source_list = ListProperty([])  # selectable list of audio files
    sourceProperty = StringProperty()  # label showing the currently active audio input
    predictorProperty = StringProperty()  # label showing the currently active predictor

    def start_Button_pressed(self, label):
        self.start_button.disabled = True
        # get current spectrogram and predictions periodically by polling
        Clock.schedule_interval(App.get_running_app().getCurrentSpectrogram, UPDATE_RATE)
        Clock.schedule_interval(App.get_running_app().getCurrentPrediction, UPDATE_RATE)

    def liveOrFileSettingHasChanged(self, instance, value):
        App.get_running_app().setIsLive(value)

    def predSettingHasChanged(self, *args):
        App.get_running_app().setPredictor(args[0].selection[0].text)

    def sourceSettingHasChanged(self, *args):
        App.get_running_app().setFile(args[0].selection[0].text)

    @mainthread
    def update_Spectrogram_Image(self, image):
        image_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='rgb')
        arr = image.flatten()
        image_texture.blit_buffer(arr, colorfmt='rgb', bufferfmt='ubyte')
        self.img_texture.texture = image_texture

    @mainthread
    def update_Class_Prob_Bar(self, label, width, height, index):
        self.class_list[index] = label
        self.prob_list[index] = width
        self.class_bar_height[index] = height


class MainApp(App):

    kv_directory = 'viewer/kv'

    screen_width, screen_heigth = getScreenResolution()
    window_width, window_heigth = screen_width / 1.5, screen_heigth / 1.4
    Window.size = (window_width, window_heigth)

    def build(self):
        self.window = AudioTaggerWindow()
        self.window.pred_list = self.loadPredictors()
        self.window.source_list = self.loadSources()

        # init
        self.isLive = True
        self.window.sourceView.height = 0.0
        self.window.fileMessageLabel.height = self.window.fileMessageLabel.parent.height * 0.8
        self.file = self.window.source_list[0]['displayname']
        self.predictor = self.window.pred_list[0]['displayname']

        self.setSummaryLabels()

        # register trigger functions to update new settings (predictors, input source)
        self.window.input_switch.bind(active=self.window.liveOrFileSettingHasChanged)
        self.window.predView.adapter.bind(on_selection_change=self.window.predSettingHasChanged)  # doesn't work in .kv file
        self.window.sourceView.adapter.bind(on_selection_change=self.window.sourceSettingHasChanged)  # doesn't work in .kv file

        return self.window

    def setSummaryLabels(self):
        if self.isLive:
            self.window.sourceProperty = 'Microphone'
        else:
            self.window.sourceProperty = self.file
        self.window.predictorProperty = self.predictor

    def loadPredictors(self):
        response = urlopen(SERVER_ADR + "/pred_list")
        return json.loads(response.read())

    def loadSources(self):
        response = urlopen(SERVER_ADR + "/audiofile_list")
        return json.loads(response.read())

    def setIsLive(self, value):
        self.isLive = value

        # display file chooser only if mic mode is off
        if self.isLive:
            self.window.sourceView.height = 0.0
            self.window.fileMessageLabel.height = self.window.fileMessageLabel.parent.height * 0.5
        else:
            self.window.sourceView.height = self.window.sourceView.parent.height * 0.8
            self.window.fileMessageLabel.height = 0.0

        self.notifyBackendAboutSettingsChanged()

    def setFile(self, value):
        self.file = value
        self.notifyBackendAboutSettingsChanged()

    def setPredictor(self, value):
        self.predictor = value
        self.notifyBackendAboutSettingsChanged()

    def getCurrentSpectrogram(self, dt):
        response = urlopen(SERVER_ADR + "/live_visual")

        image = np.fromstring(response.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.flip(image, 0)

        # image = cv2.resize(image, (image.shape[1]*3, image.shape[0]*3))

        # send new spectrogram to UI thread
        self.window.update_Spectrogram_Image(image)

    def getCurrentPrediction(self, dt):
        response = urlopen(SERVER_ADR + "/live_pred")
        prob_list = json.loads(response.read())
        # ordered by class probability if n of classes < 5, else stable position
        if len(prob_list) > 5:
            prob_list = sorted(prob_list, key=lambda i: i[1], reverse=True)
        for i in range(5):
            if i < len(prob_list):
                class_label = prob_list[i][0]
                class_width = prob_list[i][1] * self.window.class1Label.parent.width
                # send new predictions to UI thread
                self.window.update_Class_Prob_Bar(class_label, class_width, 20, i)
            else:
                # send new predictions to UI thread
                self.window.update_Class_Prob_Bar('', 1, 1, i)

    def notifyBackendAboutSettingsChanged(self):
        self.setSummaryLabels()
        fileId = [elem['id'] for elem in self.window.source_list if elem['displayname'] == self.file][0]
        predictorId = [elem['id'] for elem in self.window.pred_list if elem['displayname'] == self.predictor][0]
        settingsDict = {'isLive': 1 if self.isLive else 0, 'file': fileId, 'predictor': predictorId}
        # send new settings to backend
        res = requests.post(SERVER_ADR + '/settings', json=settingsDict)

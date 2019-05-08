import numpy as np
import pickle


class MASPModel:

    def __init__(self, applause_detector, music_detector, speech_detector):

        self.applause_detector = applause_detector
        self.music_detector = music_detector
        self.speech_detector = speech_detector

    def predict_proba(self, x):

        cfa, cft, llf = x

        if len(cfa.shape) !=2:
            cfa = np.expand_dims(cfa, 0)

        app_prob = self.applause_detector.predict_proba(llf)[:, 1]
        music_prob = self.music_detector.predict_proba(np.hstack([llf, cfa]))[:, 1]
        speech_prob = self.speech_detector.predict_proba(np.hstack([llf, cft]))[:, 1]

        return np.asarray([app_prob, music_prob, speech_prob]).T

    def predict(self, x):

        probs = self.predict_proba(x)
        probs[probs < 0.5] = 0
        probs[probs >= 0.5] = 1

        return np.asarray(probs, dtype=np.int32)


def load_masp_model(applause_det_path, music_det_path, speech_det_path):
    applause_model = pickle.load(open(applause_det_path, 'rb'))
    speech_model = pickle.load(open(speech_det_path, 'rb'))
    music_model = pickle.load(open(music_det_path, 'rb'))

    return MASPModel(applause_model, music_model, speech_model)

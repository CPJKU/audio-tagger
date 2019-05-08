import pickle
import os
import numpy as np
import madmom

from server.consumer.predictors.masp_predictor.masp_model import load_masp_model
from server.consumer.predictors.predictor_contract import PredictorContract
from server.config.config import PROJECT_ROOT, BUFFER_SIZE, SAMPLE_RATE
from collections import deque
from threading import Thread, Event
from server.consumer.predictors.masp_predictor.features import get_spec, low_level_features, cft, cfa, get_cent_conversion_matrix

class PredictionThread(Thread):
    """
    Thread for periodically computing new class predictions
    based on the currently available sliding window.

    Attributes
    ----------
    provider : PredictorContract
        reference to the predictor the thread belongs to
    _stopevent : threading.Event
        indicator for stopping a thread loop

    Methods
    -------
    run()
        method triggered when start() method is called.

    join()
        sends stop signal to thread.
    """
    def __init__(self, provider, name='PredictionThread'):
        """
        Parameters
        ----------
        provider : PredictorContract
            reference to the predictor the thread belongs to
        name : str
            the name of the thread
        """
        self.provider = provider
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        """Periodically computes new predictions based on
        the currently available sliding window. After each iteration
        the method informs ``AudioTaggerManager`` about the new
        predictions.
        """
        while not self._stopevent.isSet():
            if len(self.provider.manager.sharedMemory) > 0:   # start consuming once the producer has started
                probs = self.provider.manager.predProvider.predict()
                self.provider.manager.onNewPredictionCalculated(probs)

    def join(self, timeout=None):
        """Stops the thread.

        This method tries to stop a thread. When timeout has passed
        and the thread could not be stopped yet, the program continues.
        If timeout is set to None, join blocks until the thread is stopped.

        Parameters
        ----------
        timeout : float
            a timeout value in seconds

        """
        self._stopevent.set()
        Thread.join(self, timeout)


class MASPPredictor(PredictorContract):
    """
    Implementation of a PredictorContract. This class
    serves as a dummy predictor and shows the basic
    structure of a predictor.

    Attributes
    ----------
    predThread:
        reference pointing to the prediction thread

    Methods
    -------
    start()
       starts all necessary sub tasks of this predictor.
    stop()
       stops all necessary sub tasks of this predictor.
    predict()
       dummy prediction returning random class probabilities.
    """

    def __init__(self):

        self.classes = ["Applause", "Music", 'Speech']


        predictor_path = os.path.join(PROJECT_ROOT,'server/consumer/predictors/masp_predictor/')
        self.model = load_masp_model(os.path.join(predictor_path, 'applause_detector'),
                                     os.path.join(predictor_path, 'music_detector'),
                                     os.path.join(predictor_path, 'speech_detector'))
        # self.applause_model = pickle.load(open(os.path.join(PROJECT_ROOT,
        #                             'server/consumer/predictors/masp_predictor/applause_detector'), 'rb'))
        # self.speech_model = pickle.load(open(os.path.join(PROJECT_ROOT,
        #                             'server/consumer/predictors/masp_predictor/speech_detector'), 'rb'))
        # self.music_model = pickle.load(open(os.path.join(PROJECT_ROOT,
        #                             'server/consumer/predictors/masp_predictor/music_detector'), 'rb'))

        self.CM_norm = get_cent_conversion_matrix(4096, SAMPLE_RATE)
        self.lastProceededGroundTruth = -1

        self.probs_median = deque(maxlen=3)
        self.probs = [[elem, 0, index] for index, elem in enumerate(self.classes)]

    def start(self):
        """Start all sub tasks necessary for continuous prediction.
        """
        self.predThread = PredictionThread(self)
        self.predThread.start()

    def stop(self):
        """Stops all sub tasks
        """
        self.predThread.join()

    def predict(self):
        """dummy predictor
        """
        t = self.manager.tGroundTruth

        # if t > 16 and t >= self.lastProceededGroundTruth + 3:
        if t > 16 and t != self.lastProceededGroundTruth:

            # collect last 16*1024/sample_rate ms

            sig = []
            for i in range(t-16, t):
                frame = self.manager.sharedMemory[i % BUFFER_SIZE]  # modulo avoids index under/overflow

                frame = np.fromstring(frame, np.int16)
                sig.append(frame)

            sig = np.hstack(sig)

            spec = get_spec(sig)

            low_feature = low_level_features(sig=sig, spec=spec, sample_rate=SAMPLE_RATE)
            cft_feature = cft(spec=spec, CM_norm=self.CM_norm, sample_rate=SAMPLE_RATE)
            cfa_feature = cfa(spec=spec)

            self.probs_median.append(self.model.predict_proba((cfa_feature, cft_feature, low_feature))[0])

            probs = np.median(self.probs_median, axis=0)

            # somehow JSON does not like np.float32 for serialization
            probs = np.asarray(probs, dtype=float)

            self.probs = [[elem, probs[index], index] for index, elem in enumerate(self.classes)]

            self.lastProceededGroundTruth = t

        return self.probs


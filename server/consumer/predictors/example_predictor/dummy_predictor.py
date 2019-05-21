"""Dummy predictor which serves as a template implementation
for a new predictor. At the end of each prediction iteration
it is essential to call the method
``onNewPredictionCalculated(probs)`` of ``AudioTaggerManager``
and send it the new predictions.

"""
import random

from threading import Thread, Event

from server.consumer.predictors.predictor_contract import PredictorContract


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


class DummyPredictor(PredictorContract):
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
    def start(self):
        """Start all sub tasks necessary for continuous prediction.
        """
        self.predThread = PredictionThread(self)
        self.predThread.start()

    def stop(self):
        """Stops all sub tasks
        """
        try:
            self.predThread.join()
        except:
            print("Join call on a non existing thread is ignored...")

    def predict(self):
        """dummy predictor
        """
        probs = [[elem, random.uniform(0, 1), index] for index, elem in enumerate(["Class 1", "Class 2", "Class 3"])]
        return probs
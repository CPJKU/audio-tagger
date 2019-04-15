import Queue
# from collections import deque

from server.config.config import BUFFER_SIZE

class IPredictor:

    def __init__(self):
        # self.buffer = deque(maxlen=BUFFER_SIZE)
        self.buffer = Queue.Queue(maxsize=BUFFER_SIZE)
        
    def refreshBuffer(self):
        self.buffer = Queue.Queue(maxsize=BUFFER_SIZE)
        
    def predict(self):
        """
        Executes a particular predictor model

        This function is the wrapper called by the prediction thread in the audio tagger model.

        Parameters
        ----------

        Returns
        -------

        Notes
        ------
        At the end of the prediction,
        self.model.onNewPredictionCalculated(probabilityArray) is essential to inform the model
        about new prediction.

        """
        raise NotImplementedError


import os
import numpy as np

from server.consumer.predictors.masp_nn_predictor.masp_nn_model import load_masp_nn_model
from server.consumer.predictors.predictor_contract import PredictorContract
from server.config.config import PROJECT_ROOT, BUFFER_SIZE, SAMPLE_RATE
from collections import deque
from server.consumer.predictors.masp_predictor.features import get_spec, low_level_features, cft,\
    cfa, get_cent_conversion_matrix
from server.consumer.predictors.prediction_thread import PredictionThread


class MASPNNPredictor(PredictorContract):
    """
    Implementation of a PredictorContract. This class
    uses a simple neural network to simultaneously predict whether applause,
    music or speech is present in the current audio.

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
        predictor_path = os.path.join(PROJECT_ROOT, 'server/consumer/predictors/masp_nn_predictor/')
        self.model = load_masp_nn_model(os.path.join(predictor_path, 'masp_nn_predictor.pt'),
                                        os.path.join(predictor_path, 'masp_nn_stats.npz'))

        self.predThread = None
        # Hz to cent conversion matrix used for calculating the CFT feature
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

        t = self.manager.tGroundTruth

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


"""This module contains a prediction consumer and therefore
inherits from ``PredictorContract``. It further defines two Thread
classes ``SlidingWindowThread`` and ``PredictionThread``.

It takes
audio chunks from the shared memory of ``AudioTaggerManager`` based
on the current global timestamp ``tGroundTruth`` and calculates a prediction
based on a spectrogram computation from the past 256 audio chunks.
Due to performance issues, the computations are cached and only the
audio chunk indicated by ``tGroundTruth`` is computed newly by a
separate Thread (SlidingWindowThread). Finally, this produces a cached spectrogram as a
sliding window over time.
The second Thread (PredictionThread) periodically access the current sliding window,
computes a class prediction with a pre-trained convolutional neural network based
on the current spectrogram as input. Finally, ``AudioTaggerManager`` is informed
about the new predictions. Therefore it is essential to call the method
``onNewPredictionCalculated(probs)`` of ``AudioTaggerManager`` and send it the
new predictions.


"""
import os
import numpy as np
import torch

from scipy import sparse
from threading import Thread, Event
from server.consumer.predictors.prediction_thread import PredictionThread
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilterbank
from madmom.processors import SequentialProcessor

from server.config.config import PROJECT_ROOT, BUFFER_SIZE, SAMPLE_RATE
from server.consumer.predictors.predictor_contract import PredictorContract
from server.consumer.predictors.dcase_predictor.baseline_model import load_dcase_model


DCASE_CLASSES = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime",
                 "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close",
                 "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong",
                 "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow",
                 "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine",
                 "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]


def fft_frequencies(num_fft_bins, sample_rate):
    return np.fft.fftfreq(num_fft_bins * 2, 1. / sample_rate)[:num_fft_bins]


class SlidingWindowThread(Thread):
    """
    Thread for processing new audio chunks, computes its
    spectrogram representation and appends it to the cached
    sliding window.

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
    def __init__(self, provider, name='SlidingWindowThread'):
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
        """Periodically computes sliding windows
        """
        while not self._stopevent.isSet():
            if len(self.provider.manager.sharedMemory) > 0: # start consuming once the producer has started
                self.provider.compute_spectrogram()

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


class DcasePredictor(PredictorContract):
    """
    Implementation of a PredictorContract. This class
    makes predictions where spectrograms are considered
    as inputs and a convolutional neural network produces
    class probabilities.

    Attributes
    ----------
    sig_proc : madmom.Processor
        processor which outputs sampled audio signals
    fsig_proc : madmom.Processor
        processor which produces overlapping frames based on sampled signals
    spec_proc : madmom.Processor
        processor which computes a spectrogram with stft based on framed signals
    filt_proc : madmom.Processor
        processor which filters and scales a spectrogram
    processorPipeline : SequentialProcessor
        creates pipeline of elements of type madmom.Processor
    classes : list of str
        class list
    device : str
        indicates the processor to be used for neural network prediction
    prediction_model : baseline_net.Net
        holds a reference to the CNN architecture
    sliding_window : 2d numpy array
        cache for previously calculated spectrograms
    lastProceededGroundTruth : int
        variable to keep track of the last processed audio chunk
    slidingWindowThread:
        reference pointing to the sliding window thread
    predictionThread:
        reference pointing to the prediction thread

    Methods
    -------
    start()
       starts all necessary sub tasks of this predictor.
    stop()
       stops all necessary sub tasks of this predictor.
    computeSpectrogram()
       compute a spectrogram based on the most current audio chunk.
    predict()
       CNN prediction based on current spectrogram input.
    """

    def __init__(self):
        """
        Parameters
        ----------
        prediction_model : baseline_net.Net
           holds a reference to the CNN architecture
        sliding_window : 2d numpy array
           cache for previously calculated spectrograms
        lastProceededGroundTruth : int
           variable to keep track of the last processed audio chunk
        """
        # load model with its tuned weight parameters
        self.model = load_dcase_model(os.path.join(PROJECT_ROOT,
                                    'server/consumer/predictors/dcase_predictor/baseline_net.pt'))

        # sliding window as cache
        self.sliding_window = np.zeros((128, 256), dtype=np.float32)
        self.lastProceededGroundTruth = None

        # madmom pipeline for spectrogram calculation
        sig_proc = SignalProcessor(num_channels=1, sample_rate=32000, norm=True)
        fsig_proc = FramedSignalProcessor(frame_size=1024, hop_size=128, origin='future')
        spec_proc = SpectrogramProcessor(frame_size=1024)

        self.processor = SequentialProcessor([sig_proc, fsig_proc, spec_proc])

        self.slidingWindowThread = None
        self.predictionThread = None

        self.dense_filterbank = np.array(LogarithmicFilterbank(
            fft_frequencies(512, SAMPLE_RATE),
            num_bands=26,
            fmin=20,
            fmax=14000,
            fref=440,
            norm_filters=True,
            unique_filters=True,
            bands_per_octave=True
        ))

        self.sparse_filterbank = sparse.csr_matrix(self.dense_filterbank)

    def start(self):
        """Start all sub tasks necessary for continuous prediction.
        """
        self.slidingWindowThread = SlidingWindowThread(self)
        self.predictionThread = PredictionThread(self)
        self.slidingWindowThread.start()
        self.predictionThread.start()

    def stop(self):
        """Stops all sub tasks
        """

        try:
            self.slidingWindowThread.join()
            self.predictionThread.join()
        except:
            print("Join call on a non existing thread is ignored...")

        del self.model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_spectrogram(self):
        """This methods first access the global time variable ``tGroundTruth``
        and reads audio chunk the time variable points to. Afterwards, the defined
        madmom pipeline is processed to get the spectrogram representation of the
        single chunk. Finally, the sliding window is updated with the new audio chunk.
        """

        t = self.manager.tGroundTruth
        # if thread faster than producer, do not consume same chunk multiple times
        if t != self.lastProceededGroundTruth:
            frame = self.manager.sharedMemory[(t - 1) % BUFFER_SIZE]   # modulo avoids index under/overflow
            frame = np.fromstring(frame, np.int16)
            spectrogram = self.processor.process(frame)

            frame = spectrogram[0]

            frame = sparse.csr_matrix.dot(frame, self.sparse_filterbank)
            frame = np.log10(frame + 1)
            if np.any(np.isnan(frame)):
                frame = np.zeros_like(frame, dtype=np.float32)

            # update sliding window
            self.sliding_window[:, 0:-1] = self.sliding_window[:, 1::]
            self.sliding_window[:, -1] = frame

            self.lastProceededGroundTruth = t

    def predict(self):
        """ This method executes the actual prediction task based on the
        currently available slinding window. The sliding window is sent
        into the CNN model and the correpsonding softmax output for the
        respecive classes are returned

        Returns
        -------
        probs : array of list objects
            an array of number of classes entries where each entry consists of
            the class name, its predicted probability and a position index.
            Example:
            ``[["class1", 0.0006955251446925104, 0], ["class2", 0.0032770668622106314, 1], ...]``
        """

        input = self.sliding_window[np.newaxis, np.newaxis]
        predictions = self.model.predict_proba(input)
        probs = [[elem, predictions[index].item(), index] for index, elem in enumerate(DCASE_CLASSES)]
        return probs

"""This module implements a consumer which takes audio input
chunks and computes the corresponding spectrogram for
visual representation. It takes audio chunks from the shared
memory in ``AudioTaggerManager`` based on the global timing
variable ``tGroundTruth``.
Due to performance issues, the computations are cached and only the
audio chunk indicated by ``tGroundTruth`` is computed newly by a
separate Thread (SlidingWindowThread). Finally, this produces a cached spectrogram as a
sliding window over time.
Finally the method ``onNewVisualisationCalculated(spec)`` informs the ``AudioTaggerManager``
that a new spectrogram is available.
"""

import numpy as np

from threading import Thread, Event

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.audio.filters import LogFilterbank
from madmom.processors import SequentialProcessor

from server.config.config import BUFFER_SIZE
from server.consumer.visualizers.visualisation_contract import VisualisationContract


class VisualisationThread(Thread):
    """
       Thread for processing new audio chunks, computes its
       spectrogram representation and appends it to the cached
       sliding window.

       Attributes
       ----------
       provider : VisualisationContract
           reference to the visualizer the thread belongs to
       _stopevent : threading.Event
           indicator for stopping a thread loop

       Methods
       -------
       run()
           method triggered when start() method is called.

       join()
           sends stop signal to thread.
       """
    def __init__(self, provider, name='VisualisationThread'):
        """
        Parameters
        ----------
        provider : PredictorContract
            reference to the visualizer the thread belongs to
        name : str
            the name of the thread
        """
        self.provider = provider
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        """Periodically computes sliding windows. At the end
        of each iteration, the manager is informed that a new
        spectrogram has been computed.
        """
        while not self._stopevent.isSet():
            if len(self.provider.manager.sharedMemory) > 0: # start consuming once the producer has started
                spec = self.provider.compute_spectrogram()
                self.provider.manager.onNewVisualisationCalculated(spec)

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


class MadmomSpectrogramProvider(VisualisationContract):
    """
    Implementation of a VisualisationContract. This class
    computes new spectrograms based on the most current
    audio chunks which is indicated via ``tGroundTruth``.

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
    sliding_window : 2d numpy array
        cache for previously calculated spectrograms
    lastProceededGroundTruth : int
        variable to keep track of the last processed audio chunk
    visThread:
        reference pointing to the sliding window thread

    Methods
    -------
    start()
       starts all necessary sub tasks of this visualizer.
    stop()
       stops all necessary sub tasks of this visualizer.
    computeSpectrogram()
       compute a spectrogram based on the most current audio chunk.
    """

    # madmom pipeline for spectrogram calculation
    sig_proc = SignalProcessor(num_channels=1, sample_rate=32000, norm=True)
    fsig_proc = FramedSignalProcessor(frame_size=1024, hop_size=128, origin='future')
    spec_proc = SpectrogramProcessor(frame_size=1024)
    filt_proc = LogarithmicFilteredSpectrogramProcessor(filterbank=LogFilterbank, num_bands=26, fmin=20, fmax=14000)
    processorPipeline = SequentialProcessor([sig_proc, fsig_proc, spec_proc, filt_proc])

    def __init__(self):
        """
        Parameters
        ----------
        sliding_window : 2d numpy array
           cache for previously calculated spectrograms
        lastProceededGroundTruth : int
           variable to keep track of the last processed audio chunk
        """

        # sliding window as cache
        self.sliding_window = np.zeros((128, 256), dtype=np.float32)
        self.lastProceededGroundTruth = None

    def start(self):
        """Start all sub tasks necessary for continuous spectrograms.
        """
        self.visThread = VisualisationThread(self)
        self.visThread.start()

    def stop(self):
        """Stops all sub tasks
        """
        try:
            self.visThread.join()
        except:
            print("Join call on a non existing thread is ignored...")

    def compute_spectrogram(self):
        """This methods first access the global time variable ``tGroundTruth``
        and reads audio chunk the time variable points to. Afterwards, the defined
        madmom pipeline is processed to get the spectrogram representation of the
        single chunk. Finally, the sliding window is updated with the new audio chunk
        and a copy of the sliding window is returned to the calling thread.

        Returns
        -------
        sliding_window : 2d numpy array of float values
            returns a copy of the current sliding window spectrogram
        """
        # if thread faster than producer, do not consume same chunk multiple times
        t = self.manager.tGroundTruth
        if t != self.lastProceededGroundTruth:
            frame = self.manager.sharedMemory[(t - 1) % BUFFER_SIZE]   # modulo avoids index under/overflow
            frame = np.fromstring(frame, np.int16)
            spectrogram = self.processorPipeline.process(frame)

            frame = spectrogram[0]
            if np.any(np.isnan(frame)):
                frame = np.zeros_like(frame, dtype=np.float32)

            # update sliding window
            self.sliding_window[:, 0:-1] = self.sliding_window[:, 1::]
            self.sliding_window[:, -1] = frame

            self.lastProceededGroundTruth = t

        return self.sliding_window.copy()



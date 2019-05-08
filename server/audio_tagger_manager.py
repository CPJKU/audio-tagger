"""Audio Tagger's main module which manages audio producers and all consumers.

In this central module of the backend system, we define major components
like a shared memory implemented as a ring buffer which holds audio chunks
coming from the producers (MicrophoneThread and AudiofileThread). Respectively,
all the chunks are consumed by modules which e.g. compute spectrogram
representation or make predictions based on a trained model for an
audio input chunk.
Once there are multiple consumers, synchronization needs to be considered.
To ensure that consumers do not drift off too far, the audio tagger manager
keeps track of a global timing variable ``tGroundTruth``. It is increased
every time a producer delivers a new audio chunk. When a consumer is ready
to process new chunks, it asks for ``tGroundTruth``'s timestamp and reads
chunks from shared memory related to this timestamp.

"""

import wave
import pyaudio
import numpy as np
from collections import deque

from pydoc import locate
from threading import Thread, Event

from server.config.config import BUFFER_SIZE, START_FILE, CHUNK_SIZE, SAMPLE_RATE, N_CHANNELS


class MicrophoneThread(Thread):
    """
    Thread for audio input over microphone.

    Attributes
    ----------
    manager : audio tagger manager object
        reference to the audio tagger manager
    p : PyAudio object
        an object to read audio input and process
        it to get digital chunks of samples.
    stream : PyAudio stream object
        stream of the pyaudio object
    _stopevent : threading.Event
        indicator for stopping a thread loop

    Methods
    -------
    run()
        method triggered when start() method is called.

    join()
        sends stop signal to thread.
    """
    def __init__(self, manager, name='MicrophoneThread'):
        """
        Parameters
        ----------
        manager : audio tagger manager object
            reference to the audio tagger manager
        name : str
            the name of the thread
        """
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=N_CHANNELS,
                                  rate=SAMPLE_RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK_SIZE)
        self.manager = manager
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        """Reads sampled audio chunks coming from a microphone and
        puts them into the shared memory. Finally, the global time variable
        ``tGroundTruth`` is incremented by 1.
        """

        while not self._stopevent.isSet():
            chunk = self.stream.read(CHUNK_SIZE)
            self.manager.putToSM(chunk)   # insert new chunk into shared memory
            self.manager.tGroundTruth += 1    # increment global timestamp variable
            self.manager.tGroundTruth = self.manager.tGroundTruth % BUFFER_SIZE # ring buffer implementation

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


class AudiofileThread(Thread):
    """
    Thread for audio input over files.

    Attributes
    ----------
    manager : audio tagger manager object
        reference to the audio tagger manager
    p : PyAudio object
        an object to read audio input and process
        it to get digital chunks of samples
    wf : wave object
        opens a wave object which reads a audio file
        with path ``filePath``
    stream : PyAudio stream object
        stream of the pyaudio object
    _stopevent : threading.Event
        indicator for stopping a thread loop

    Methods
    -------
    run()
        method triggered when start() method is called.

    join()
        sends stop signal to thread.
    """
    def __init__(self, manager, filePath, name='AudiofileThread'):
        """
        Parameters
        ----------
        manager : audio tagger manager object
            reference to the audio tagger manager
        filePath : str
            path to the selected audio file used for input
        name : str
            the name of the thread
        """

        self.p = pyaudio.PyAudio()
        self.wf = wave.open(filePath, 'rb')
        self.stream = self.p.open(format=pyaudio.paInt16,
                channels=N_CHANNELS,
                rate=SAMPLE_RATE,
                output=True)
        self.manager = manager
        self._stopevent = Event()
        Thread.__init__(self, name=name)

    def run(self):
        """Reads sampled audio chunks coming from an audio file and
        puts them into the shared memory. Finally, the global time variable
        ``tGroundTruth`` is incremented by 1.
        """
        chunk = self.wf.readframes(CHUNK_SIZE)
        while not self._stopevent.isSet() and chunk != b'':
            self.stream.write(chunk)
            self.manager.putToSM(chunk)   # insert new chunk into shared memory
            self.manager.tGroundTruth += 1    # increment global timestamp variable
            self.manager.tGroundTruth = self.manager.tGroundTruth % BUFFER_SIZE # ring buffer implementation
            chunk = self.wf.readframes(CHUNK_SIZE)

        self.stream.stop_stream()
        self.stream.close()

        self.p.terminate()

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


class AudioTaggerManager:
    """
    This is the central management class of the audio tagger backend system.

    Attributes
    ----------
    visualProvider : VisualisationContract
        a consumer which processes audio chunks to visual representation
    predProvider : PredictorContract
        a consumer which processes audio chunks to class predictions
    predList : list
        list of available predictors
    audiofileList : list
        list of available audio files
    curVisual : 2d numpy array of float values
        holds the current visual representation object
    curPred : numpy array of list objects
        holds the current class prediction object
    sharedMemory : deque
        shared memory object (ring buffer) holding audio chunks
    tGroundTruth: int
        global timing variable used for synchronization

    Methods
    -------
    getPredList()
        returns a list of available predictors.
    getAudiofileList()
        returns a list of available audio files.
    setPredProvider()
        set a new prediciton provider of type ``PredictorContract``.
    getVisualisation()
        returns the most recent visualisation.
    getPrediction()
        returns the most recent class predictions.
    onNewVisualisationCalculated(image)
        called from visualisation consumers when new representation
        is computed.
    onNewPredictionCalculated(prob_dict)
        called from predictor consumers when new class predictions
        are computed.
    startThreads()
        start producers and consumers when audio tagger manager has
        completed its initialization.
    refreshAudioTagger()
        method is called when frontend informs backend about changed
        settings regarding predictors and audio input.
    putToSM()
        adds a new audio chunk to the shared memory.
    """
    def __init__(self, visualProvider, predProvider, predList, audiofileList):
        """
        Parameters
        ----------
        visualProvider : VisualisationContract
            consumer which processes visual representation
        predProvider : PredictorContract
            consumer which processes class predictions
        predList : list
            list of available predictors
        audiofileList : list
            list of available audio files
        """
        self.visProvider = visualProvider
        self.predProvider = predProvider

        # initialization of visualization and prediction output
        self.curVisual = np.zeros((128, 256), dtype=np.float32)
        self.curPred = [["Class{}".format(index), 0.2, index] for index in range(10)]

        # consumers inform manager if an audio chunk is processed
        self.visProvider.registerManager(self)
        self.predProvider.registerManager(self)

        # selectable audio files and predictors
        self.predList = predList
        self.audiofileList = audiofileList

        self.sharedMemory = deque(maxlen=BUFFER_SIZE)

        self.tGroundTruth = 0   # global timestamp to keep up synchronization of consumers

        self.startThreads()

    def getPredList(self):
        """Gets the list of predictors

        Returns
        -------
        list
            a list of available predictors
        """
        return self.predList

    def getAudiofileList(self):
        """Gets the list of audio files

        Returns
        -------
        list
            a list of available audio files
        """
        return self.audiofileList

    def setPredProvider(self, predProvider):
        """set the reference of the currently active predictor object

        Parameters
        ----------
        predProvider : PredictorContract
            consumer which processes class predictions

        """
        self.predProvider = predProvider

    def getVisualisation(self):
        """Gets the newest visual representation processed
        by backend.

        Returns
        -------
        curVisual : 2d numpy array of float values
            returns a copy of the current visual representation object
        """
        return self.curVisual.copy()

    def getPrediction(self):
        """Gets the newest class prediction processed
        by backend.

        Returns
        -------
        curPred : numpy array of list objects
            returns a copy of the current class prediction object
        """
        return self.curPred.copy()

    def onNewVisualisationCalculated(self, image):
        """Is called every time a visualisation consumer
        has processed a new visual representation item.

        Parameters
        ----------
        image : 2d numpy array of float values
            holds the current visual representation object
        """
        self.curVisual = image

    def onNewPredictionCalculated(self, prob_dict):
        """Is called every time a predictor consumer
        has processed a new class prediction item.

        Parameters
        ----------
        prob_dict : numpy array of list objects
            holds the current class prediction object
        """
        self.curPred = prob_dict

    def startThreads(self):
        """start producers and consumers when audio tagger manager has
        completed its initialization.
        """

        # decide if audio input comes from microphone or file
        if START_FILE == None:
            self.producerThread = MicrophoneThread(self)
        else:
            filePath = [elem['path'] for elem in self.getAudiofileList() if elem['id'] == START_FILE][0]
            self.producerThread = AudiofileThread(self, filePath)

        self.producerThread.start()

        # start consumers
        self.visProvider.start()
        self.predProvider.start()

    # Refresh function
    # This function is called when the frontend
    # changes audio mode, file or predictor
    def refreshAudioTagger(self, settings):
        """method is called when frontend informs backend about changed
        settings regarding predictors and audio input.

        Parameters
        ----------
        settings : dictionary
            a dictionary holding the a flag for microphone/audio file input
            and IDs for the desired predictor and audio file, respectively.

        """

        # stop producer and consumers
        self.producerThread.join()

        self.visProvider.stop()
        self.predProvider.stop()

        # reset ring buffer
        self.tGroundTruth = 0
        self.sharedMemory.clear()

        # restart audio tagger with delivered settings
        isLive = settings['isLive']
        file = settings['file']
        predictor = settings['predictor']

        # decide if audio input comes from microphone or file
        if isLive:
            self.producerThread = MicrophoneThread(self)
        else:
            filePath = [elem['path'] for elem in self.getAudiofileList() if elem['id'] == file][0]
            self.producerThread = AudiofileThread(self, filePath)

        # load selected prediction class via reflection, update references for the new predictor object
        predictorClassPath = [elem['predictorClassPath'] for elem in self.getPredList() if elem['id'] == predictor][0]
        predProviderClass = locate('server.consumer.predictors.{}'.format(predictorClassPath))
        newPredProvider = predProviderClass()
        self.setPredProvider(newPredProvider)
        self.predProvider.registerManager(self)

        # restart producer and consumers
        self.producerThread.start()

        self.visProvider.start()
        self.predProvider.start()

    def putToSM(self, chunk):
        """adds a new audio chunk to the shared memory.

        Parameters
        ----------
        chunk : bytes
            an array of audio sample values encoded as byte string

        """

        # first iteration of ring buffer to avoid IndexOutOfRangeError
        if len(self.sharedMemory) < BUFFER_SIZE:
            self.sharedMemory.append(chunk)
        else:
            self.sharedMemory[self.tGroundTruth] = chunk

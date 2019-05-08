""" This module should declare a contract for
all consumers which act as a predictor.

"""


class VisualisationContract:
    """ Contract class for all predictor consumers

    Attributes
    ----------
    manager : AudioTaggerManager
        reference to the audio tagger manager object

    Methods
    -------
    registerManager()
        set reference to the audio tagger manager.
        This is needed for accessing shared memory,
        global timing variable and triggering events.
    start()
        start the visualisation consumer to do work.
    stop()
        stop the visualisation consumer to do work.

    """
    def registerManager(self, manager):
        """set reference to the audio tagger manager.
        This is needed for accessing shared memory,
        global timing variable and triggering events.

        Parameters
        ----------
        manager : AudioTaggerManager
            reference to the audio tagger manager object

        """
        self.manager = manager

    def start(self):
        """start the visualisation consumer to do work.

        Raises:
            NotImplementedError: Subclasses must implement this method

        """
        raise NotImplementedError

    def stop(self):
        """stop the visualisation consumer to do work.

        Raises:
            NotImplementedError: Subclasses must implement this method

        """
        raise NotImplementedError
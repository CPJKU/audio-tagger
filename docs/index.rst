Audio-Tagger
============

The Audio-Tagger is designed as a general-purpose tool to make predictions
along with visualizations on live-audio input.
For example, we provide a DNN-based audio scene classification predictor
based on 2018's DCASE-challenge.

The system is designed as a server-client architecture.
The server is responsible for all audio IO and for performing predictions.
The audio input can either come from a WAV-file or from live microphone input.
The client can control the server and receive the predictions
through a simple REST-API.
We provide an exemplary GUI written in Kivy along with this project.
Please follow the README to setup the required Python environments.

Modules
=======

.. toctree::
    :maxdepth: 3

    backend

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

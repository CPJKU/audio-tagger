"""This module includes some options which can be set before backend startup.
At first, please set the variable ``PROJECT_ROOT`` to the absolute path of the projects root directory.

"""
import os
# absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(__file__).split(os.sep)[:-2]
PROJECT_ROOT = os.sep.join(PROJECT_ROOT)

# set the id of initial predictor and file
START_PREDICTOR = 0
START_FILE = None  # if None then microphone is used

# audio signal producer settings
CHUNK_SIZE = 1024
SAMPLE_RATE = 32000
N_CHANNELS = 1

BUFFER_SIZE = 1000   # size of ring buffer

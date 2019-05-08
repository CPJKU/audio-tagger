"""This module implements methods to read the available
predictors and audio files.

"""

import os
import csv

from server.config.config import PROJECT_ROOT


def loadPredictors():
    """Load the predictors and their properties from
    ``predictors.csv``

    Returns
    -------
    list
        a list of dictionaries of the available predictors in the
        following format:
        ``[{"id": 0, "displayname": "DCASEPredictor",
            "classes": "41", "description": "sample description for dcase"},
           {"id": 1, "displayname": "SportsPredictor",
           "classes": "3", "description": "sample description for detecting sports"}, ...]``
    """
    with open(os.path.join(PROJECT_ROOT, 'server/config/predictors.csv')) as file:
        csvReader = csv.reader(file, delimiter=';')
        header = next(csvReader, None)  # skip header
        return [{header[0]: int(line[0]),
                 header[1]: line[1],
                 header[2]: line[2],
                 header[3]: line[3],
                 header[4]: line[4]} for line in csvReader]


def loadAudiofiles():
    """Load the available audio files from
    ``audiofiles.csv``

    Returns
    -------
    list
        a list of dictionaries of the available audio files in the
        following format:
        ``[{"id": 0, "displayname": "Trumpets"}, {"id": 1, "displayname": "Song1"}, {"id": 2, "displayname": "Song2"}, ...]``
    """
    with open(os.path.join(PROJECT_ROOT, 'server/config/audiofiles.csv')) as file:
        csvReader = csv.reader(file, delimiter=';')
        header = next(csvReader, None)  # skip header
        return [{header[0]: int(line[0]),
                 header[1]: line[1],
                 header[2]: line[2]} for line in csvReader]

"""This is the python script for starting up the backend providing the necessary
REST API interface methods to access the audio input visualisations and predictions
periodically computed by the backend.

At first, available audio files and predictors are loaded and the backend is
initialized. The starting visualisation component is still fixed in this version.
The starting predictor can be configured in the config.py module by setting the
id of the predictor listed in predictors.csv. After initialization, the module
opens an application server providing various REST interface methods. The host
is fixed at http://127.0.0.1:5000.
Possible GET requests are audio input visualisation (e.g. spectrogram) and the class predictions
of a certain model based on the current audio input. Beyond reading data from the
web server, one can also send the backend that it should switch to another
predictor or should use microphone input or audio file input.

"""

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

from pydoc import locate
from flask import Flask, Response, request

from server.consumer.visualizers.spectrogram.madmom_spectrogram_provider import MadmomSpectrogramProvider
from server.audio_tagger_manager import AudioTaggerManager
from server.config.load_config import loadPredictors, loadAudiofiles
from server.config.config import START_PREDICTOR

# load configs
predictorList = loadPredictors()
audiofileList = loadAudiofiles()

visualisationProvider = MadmomSpectrogramProvider()

# load prediction class via reflection
predictionProviderClass = locate('server.consumer.predictors.{}'.format(predictorList[int(START_PREDICTOR)]['predictorClassPath']))
predictionProvider = predictionProviderClass()

model = AudioTaggerManager(visualisationProvider, predictionProvider, predictorList, audiofileList)

# audio tagger REST API functions
app = Flask(__name__)


@app.route('/live_visual', methods=['GET'])
def live_visual():
    """Http GET interface method to request most current audio visualisation
    (URI: /live_visual).

    The backend periodically computes new visual representations of the
    currently incoming audio chunks. This method provides access to
    the most recent visual representation (e.g. spectrogram).

    Note
    ----
    In general, the method would return the same representation until
    a new one has been computed.

    Returns
    -------
    Response
        a response object with the visualisation in jpeg-format as content.
    """
    content = model.getVisualisation()
    content = convertSpecToJPG(content)
    return Response(content,
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live_visual_browser', methods=['GET'])
def live_visual_browser():
    """Http GET interface method to request most current audio visualisation
    (browser ready) (URI: /live_visual_browser).

    This method is equivalent to live_visual() except that response content
    is adapted to be visualized in the browser.

    Note
    ----
    In general, the method would return the same representation until
    a new one has been computed.

    Returns
    -------
    Response
        a response object with the visualisation in jpeg-format as content
        which can be displayed in browser.
    """
    content = model.getVisualisation()
    content = convertSpecToJPG(content)
    content = (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + content + b'\r\n\r\n')
    return Response(content,
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live_pred', methods=['GET'])
def live_pred():
    """Http GET interface method to request most current class predictions.
    (URI: /live_pred)

    Once the backend has computed new predictions based on current audio input
    they can be accessed via this REST interface method.

    Note
    ----
    In general, the method would return the same predictions until
    a new one has been computed.

    Returns
    -------
    Response : json
        a json object with the class predictions in the following form:
        ``[["Acoustic_guitar", 0.0006955251446925104, 0], ["Applause", 0.0032770668622106314, 1], ...]``

    """
    content = model.getPrediction()
    response = app.response_class(
        response=json.dumps(content),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/pred_list', methods=['GET'])
def pred_list():
    """Http GET interface method to receive a list of available predictors.
    (URI: /pred_list)

    This method returns all predictors available in the backend system.
    Each predictor comes with the following properties:

    -   ID
    -   Displayname
    -   Number of classes
    -   Description

    Note
    ----
    The ID is important since it is used to identify the desired predictor
    once a user sends a new setting to the server with send_new_settings().

    Returns
    -------
    Response : json
        a json object with the available predictors in the following form:
        ``[{"id": 0, "displayname": "DCASEPredictor", "classes": "41", "description": "sample description for dcase"},
        {"id": 1, "displayname": "SportsPredictor", "classes": "3", "description": "sample description for detecting sports"}, ...]``

    """
    content = [{'id': elem['id'],
                'displayname': elem['displayname'],
                'classes': elem['classes'],
                'description': elem['description']} for elem in model.getPredList()]

    response = app.response_class(
        response=json.dumps(content),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/audiofile_list', methods=['GET'])
def audiofile_list():
    """Http GET interface method to receive a list of available audio files.
    (URI: /audiofile_list)

    This method returns a list of audio files which can be selected
    and subsequently processed by the backend system. Each audio file
    comes with it's ID and a displayname.

    Note
    ----
    The ID is important since it is used to identify the audio file
    once a user sends a new setting to the server with send_new_settings().

    Returns
    -------
    Response : json
        a json object with the available audio files in the following form:
        ``[{"id": 0, "displayname": "Trumpets"}, {"id": 1, "displayname": "Song1"}, {"id": 2, "displayname": "Song2"}, ...]``

    """
    content = [{'id': elem['id'], 'displayname': elem['displayname']} for elem in model.getAudiofileList()]
    response = app.response_class(
        response=json.dumps(content),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/settings', methods=['POST'])
def send_new_settings():
    """Http POST interface method for sending new configuration settings
    to backend system.
    (URI: /settings)

    This methods allows to change the currently active predictor and the
    audio input source on the fly without stopping the backend. Once a user
    selected audio file input, a list of audio files is available to select
    a certain input source. The body of the POST message should look as follows:
    ``{'isLive': 1, 'file': 0, 'predictor': 1}``

    Note
    ----
    Use the same IDs for audio files and predictors as the come from
    pred_list() and audiofile_list() so the backend system can match
    the selection.

    Returns
    -------
    Http Status Code

    """
    content = request.json  # read the POST body an get the content
    model.refreshAudioTagger(content)
    return 'OK'


# Helper functions
def convertSpecToJPG(spec):
    spec = spec / 3.0
    resz_spec = 3
    spec = cv2.resize(spec, (spec.shape[1] * resz_spec, spec.shape[0] * resz_spec))
    spec = plt.cm.viridis(spec)[:, :, 0:3]
    spec_bgr = (spec * 255).astype(np.uint8)
    if spec_bgr.shape[1] < 512:
        p = (512 - spec_bgr.shape[1]) // 2
        spec_bgr = np.pad(spec_bgr, ((0, 0), (p, p), (0, 0)), mode="constant")
    spec_bgr = cv2.flip(spec_bgr, 0)
    _, curImage = cv2.imencode('.jpg', spec_bgr)

    return curImage.tobytes()


# start webserver
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False)



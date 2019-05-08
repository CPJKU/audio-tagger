"""
Script to resample given audio files to the specified sample rate in the config
"""

from server.config.config import SAMPLE_RATE, PROJECT_ROOT
import glob
import os
from scipy.io import wavfile
from subprocess import check_call

if __name__ == "__main__":

    for file in glob.glob(os.path.join(PROJECT_ROOT, 'server/files/*.wav')):

        org_sr, wf = wavfile.read(file)

        if org_sr != SAMPLE_RATE:
            print('Resample {} from {} Hz to {} Hz'.format(file, org_sr, SAMPLE_RATE))

            check_call(["ffmpeg", "-y", "-i", file, "-ar", str(SAMPLE_RATE), os.path.join(PROJECT_ROOT, 'server/files/tmp.wav')])
            os.remove(file)
            os.rename(os.path.join(PROJECT_ROOT, 'server/files/tmp.wav'), file)
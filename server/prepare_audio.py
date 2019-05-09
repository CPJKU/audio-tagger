"""
Script to resample given audio files to the specified sample rate in the config
"""

from server.config.config import SAMPLE_RATE, PROJECT_ROOT
import glob
import os
from scipy.io import wavfile
from subprocess import check_call


if __name__ == "__main__":
    PATH_FILES = os.path.join('server', 'selection')

    for file in glob.glob(os.path.join(PROJECT_ROOT, PATH_FILES, '*.wav')):
        print(file)
        path_tmpfile = os.path.join(PROJECT_ROOT, PATH_FILES, 'tmp.wav')

        # get samplerate
        org_sr, _ = wavfile.read(file)

        # normalize and clip silence
        call_sox = ['sox', file,
                    '-r', str(SAMPLE_RATE), path_tmpfile,
                    'norm', '-0.1',
                    'silence', '1', '0.025', '0.15%',
                    'norm', '-0.1',
                    'reverse',
                    'silence', '1', '0.025', '0.15%',
                    'reverse']
        check_call(call_sox)

        os.remove(file)
        os.rename(os.path.join(PROJECT_ROOT, PATH_FILES, 'tmp.wav'), file)

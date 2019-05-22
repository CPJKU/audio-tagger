import madmom
import numpy as np
from scipy import sparse
from scipy.signal import find_peaks
import librosa
from numpy.lib.stride_tricks import as_strided


CFA_FEATURE_COLUMNS = ['cfa']

CFT_FEATURE_COLUMNS = ['cgain1', 'cgain2', 'cgain3', 'cgain4', 'cgain5', 'cgain6', 'cgain7', 'cgain8', 'cgain9',
                       'cgain10', 'cgain11', 'dominant_cent_bin', 'zero_crossing_rate']

LLF_FEATURE_COLUMNS = ['spectral_centroid1', 'spectral_centroid2', 'spectral_centroid3', 'spectral_centroid4',
                       'spectral_spread1', 'spectral_spread2', 'spectral_spread3', 'spectral_spread4',
                       'spectral_flattness1', 'spectral_flattness2', 'spectral_flattness3', 'spectral_flattness4',
                       'spectral_flux1', 'spectral_flux2', 'spectral_flux3', 'spectral_flux4',
                       'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9']

EPS = 1e-8


# https://stackoverflow.com/questions/30677241/how-to-limit-cross-correlation-window-width-in-numpy
def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x


def crosscorrelation(x, y, maxlag):
    """

    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)


def get_spec(sig, fft_length=4096, fft_hop_size=1024, window=np.hanning):
    """
    Compute the magnitude spectrogram from a given raw waveform audio signal

    :param sig: raw waveform audio signal
    :param fft_length:
    :param fft_hop_size:
    :param window:
    :return: magnitude spectrogram (madmom spectrogram)
    """

    frames = madmom.audio.signal.FramedSignal(sig, frame_size=fft_length, hop_size=fft_hop_size)

    # stft
    stft = madmom.audio.stft.ShortTimeFourierTransform(frames, window=window)

    # to spectrogram
    # return madmom.audio.spectrogram.Spectrogram(stft)
    return librosa.util.normalize(madmom.audio.spectrogram.Spectrogram(stft), norm=1, axis=1)


def get_cent_conversion_matrix(fft_length, sample_rate):
    """
    Compute the matrix to convert a magnitude spectrogram from Hz to cent
    :param fft_length:
    :param sample_rate:
    :return: conversion matrix
    """
    F = madmom.audio.stft.fft_frequencies(fft_length // 2, sample_rate)
    FC = 1200 * np.log2(F + EPS)

    LC = np.arange(7500, 12025, 25)
    CM = np.zeros((FC.shape[0], LC.shape[0]), dtype=np.float32)

    t = 150
    for i, fc in enumerate(FC):
        for j, lc in enumerate(LC):
            CM[i, j] = np.abs(fc - lc) - t

    CM_masked = np.copy(CM)
    CM_masked[CM_masked > 0] = 0
    CM_boosted = CM_masked ** 2
    CM_norm = CM_boosted / CM_boosted.sum(0)

    return sparse.csr_matrix(CM_norm)


def cfa(sig=None, spec=None, binarization_threshold=0.1, nr_activation_peaks=5, nr_freq_bins=552, nnf_window_length=21,
        fft_length=4096, fft_hop_size=1024):
    """
    see: AUTOMATIC MUSIC DETECTION IN TELEVISION PRODUCTIONS
    https://pdfs.semanticscholar.org/1d57/66440b2170cf31b4df6e14d23e78f542ca67.pdf

    :param sig: raw waveform audio signal
    :param spec: magnitude spectrogram of the signal
    :param binarization_threshold:
    :param nr_activation_peaks:
    :param nr_freq_bins:
    :param nnf_window_length:
    :param fft_length:
    :param fft_hop_size:

    :return: quantified activation function
    """

    if spec is None:
        # calculate the magnitude spectrogram if none is given
        assert sig is not None
        spec = get_spec(sig, fft_length, fft_hop_size)

    spec = spec[:, :nr_freq_bins]

    # to decibel
    db_spec = 10 * np.log10(spec / 1.0 + EPS)

    # TODO: normalize ?
    norm_spec = (db_spec - db_spec.min(axis=1)[:, None]) / (db_spec.max(axis=1) - db_spec.min(axis=1) + EPS)[:, None]

    # Emphasize local peaks
    cum_sum_spec = np.cumsum(norm_spec, axis=1)
    filtered = np.zeros_like(norm_spec)

    for i in range(norm_spec.shape[1]):
        f = np.maximum(0, i - nnf_window_length // 2)
        t = np.minimum(norm_spec.shape[1] - 1, i + nnf_window_length // 2)
        l = (t - f) + 1
        filtered[:, i] = (cum_sum_spec[:, t] - (cum_sum_spec[:, f] - norm_spec[:, f])) / l
        filtered_spec = norm_spec - filtered

    # Binarization
    bin_spec = np.where(filtered_spec > binarization_threshold, 1.0, 0.0)

    # Computation of the frequency activation
    activation = bin_spec.mean(axis=0)


    # Detect strong peaks
    # TODO: can be done in a single forward pass, but lazy
    peaks, _ = find_peaks(activation)

    valleys, _ = find_peaks(1 - activation)

    pv = []

    # TODO hacky, not sure of correct
    if len(peaks) <= nr_activation_peaks or len(valleys) <= nr_activation_peaks:
        feature = 0
    else:
        if peaks[0] < valleys[0]:
            peaks = peaks[1:]
        if peaks[-1] > valleys[-1]:
            peaks = peaks[:-1]

        for i in range(len(peaks)):
            l = activation[valleys[i]]
            r = activation[valleys[i + 1]]

            h = activation[peaks[i]] - np.maximum(l, r)
            w = peaks[i] - valleys[i] if l > r else valleys[i + 1] - peaks[i]
            pv.append(h / w)

        # Quantify the continuous frequency activation
        try:
            pv = np.argpartition(pv, -nr_activation_peaks)[-nr_activation_peaks:]
        except ValueError:
            print()

        feature = activation[peaks[pv]].mean()

    return np.expand_dims(feature, 0)


def cft(sig=None, spec=None, CM_norm=None, noise_threshold=0.2, fft_length=4096, fft_hop_size=1024, sample_rate=44100):
    """
    Curved frequency trajectories

    :param sig: raw waveform audio signal
    :param spec: magnitude spectrogram of the signal
    :param CM_norm: cent conversion matrix
    :param noise_threshold:
    :param fft_length:
    :param fft_hop_size:
    :param sample_rate:
    :return:
    """

    if spec is None:
        # calculate the magnitude spectrogram if none is given
        assert sig is not None
        spec = get_spec(sig, fft_length, fft_hop_size)

    if CM_norm is None:
        # calculate the conversion matrix if none is given
        CM_norm = get_cent_conversion_matrix(fft_length, sample_rate)

    # convert magnitude spectrogram from Hz to cent
    # XC_cent = CM_norm.T @ spec.T
    XC_cent = sparse.csr_matrix.dot(spec, CM_norm).T

    # 3. Noise reduction
    min_ = np.min(XC_cent, axis=0)
    max_ = np.max(XC_cent, axis=0)
    XC = (XC_cent - min_) / (max_ - min_ + EPS)

    XC[XC <= noise_threshold] = 0

    # 4. Correlation
    offset = 3
    max_lag = 3

    c = np.zeros((2 * max_lag + 1, XC.shape[1] - offset))

    for i in range(2 + offset, XC.shape[1]):
        c[:, i - offset] = crosscorrelation(XC[:, i], XC[:, i - offset], max_lag)

    l = np.argmax(c, axis=0)
    c_max = np.max(c, axis=0)

    c_gain = np.zeros_like(c_max)
    c_gain[l > max_lag] = 1 - c[max_lag, l > max_lag] / (c_max[l > max_lag] + EPS)
    c_gain[l < max_lag] = (c[max_lag, l < max_lag] / (c_max[l < max_lag] + EPS)) - 1

    # first two entries are always zero (because of the cross correlation?)
    c_gain = c_gain[2:]

    # dominant cent bin
    S = XC.sum(axis=1)
    S_conv = np.convolve(S, [0.2, 0.2, 0.2, 0.2, 0.2])
    d = np.argmax(S_conv)

    # zero crossing rate
    zcr = np.mean(c_gain[1:] * c_gain[:-1] < 0)

    # final feature vector
    final_feature = np.concatenate((c_gain, [d], [zcr]))

    return np.expand_dims(final_feature, 0)


def low_level_features(sig=None, spec=None, fft_length=4096, fft_hop_size=1024, sample_rate=44100):
    """
    Low Level Features

    Extract the Spectral Centroid, Spectral Spread, Spectral Flattness, Spectral Flux and the first nine MFCCs
    from a given signal or spectrogram.

    :param sig: raw waveform audio signal
    :param spec: magnitude spectrogram of the signal
    :param fft_length:
    :param fft_hop_size:
    :param sample_rate:
    :return:
    """

    # MFCCs

    k = 0.97
    preprocessed_sig = sig[1:] - k*sig[:-1]

    mfcc_spec = get_spec(preprocessed_sig, fft_length, fft_hop_size, window=np.hamming)
    mel_spec = librosa.feature.melspectrogram(S=mfcc_spec.T, fmin=0, fmax=22100, n_mels=20)
    mfccs = librosa.feature.mfcc(S=mel_spec, n_mfcc=9).T

    if spec is None:
        # calculate the magnitude spectrogram if none is given
        assert sig is not None
        spec = get_spec(sig, fft_length, fft_hop_size)

    # split spectrogram into 4 subbands
    fft_frequencies = madmom.audio.stft.fft_frequencies(fft_length // 2, sample_rate)

    spec1_freq = fft_frequencies[(fft_frequencies > 129) & (fft_frequencies < 366)]
    spec1 = spec[:, (fft_frequencies > 129) & (fft_frequencies < 366)]

    spec2_freq = fft_frequencies[(fft_frequencies > 387) & (fft_frequencies < 904)]
    spec2 = spec[:, (fft_frequencies > 387) & (fft_frequencies < 904)]

    spec3_freq = fft_frequencies[(fft_frequencies > 926) & (fft_frequencies < 1981)]
    spec3 = spec[:, (fft_frequencies > 926) & (fft_frequencies < 1981)]

    spec4_freq = fft_frequencies[(fft_frequencies > 2003) & (fft_frequencies < 4134)]
    spec4 = spec[:, (fft_frequencies > 2003) & (fft_frequencies < 4134)]

    specs = [spec1, spec2, spec3, spec4]
    freqs = [spec1_freq, spec2_freq, spec3_freq, spec4_freq]

    # calculate the Spectral Centroid, Spectral Spread, Spectral Flattness and Spectral Flux
    spectral_centroid = []
    spectral_spread = []
    spectral_flattness = []
    spectral_flux = []

    for s, freq in zip(specs, freqs):

        # spectral centroid
        sc = librosa.feature.spectral_centroid(S=s.T, n_fft=fft_length, hop_length=fft_hop_size,
                                          freq=freq)[0]

        # spectral spread is also called spectral bandwidth
        # http://www.carminecella.com/teaching/Audio_features.pdf#page=12&zoom=auto,0,-437
        # could pass the centroid, but somehow yields different results?!
        ssp = librosa.feature.spectral_bandwidth(S=s.T, freq=freq, n_fft=fft_length, hop_length=fft_hop_size)[0]

        # spectral flatness
        sfm = librosa.feature.spectral_flatness(S=s.T, power=1)[0]

        # spectral flux
        sf = madmom.features.onsets.spectral_flux(s)

        spectral_centroid.append(sc)
        spectral_spread.append(ssp)
        spectral_flattness.append(sfm)
        spectral_flux.append(sf)

    spectral_centroid = np.vstack(spectral_centroid).T
    spectral_spread = np.vstack(spectral_spread).T
    spectral_flattness = np.vstack(spectral_flattness).T
    spectral_flux = np.vstack(spectral_flux).T

    features = np.concatenate((spectral_centroid, spectral_spread, spectral_flattness, spectral_flux, mfccs), axis=1)
    features = np.mean(features, axis=0)

    return np.expand_dims(features, 0)

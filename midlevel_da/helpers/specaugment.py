import os
from paths import *
import pandas as pd
import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SpecAugment:
    def __init__(self, transformations=None):
        if transformations is None:
            transformations = []
        self.transformations = transformations

    def __call__(self, spec):
        out_spec = spec
        for transform in self.transformations:
            out_spec = transform(out_spec)

        return out_spec


class SpecTransform:
    def __init__(self, transform_func, **ranges):
        self.ranges = ranges
        self.transform_func = transform_func

    def __call__(self, spec):
        params = {}
        for par, rng in self.ranges.items():
            params[par] = np.random.uniform(low=rng[0], high=rng[1])

        return self.transform_func(spec, **params)

    def __str__(self):
        ranges = ' '.join([f'{k}=({v[0]}, {v[1]})' for k, v in self.ranges.items()])
        return f"{self.transform_func.__name__} [{ranges}]"


def _get_1d_hann_filter(len, scale=1., shift=0):
    """
    Returns a 1D array with a hanning window shape.
    :param len: Length of array
    :param scale: Range (max val (1.0) - min val)
    :param shift: Samples to shift the window
    :return: 1D array
    """
    shift = int(shift)
    w = np.hanning(len)
    if shift > 0:
        w = np.pad(w, (shift, 0), mode='constant')[:-shift]
    elif shift < 0:
        w = np.pad(w, (0, np.abs(shift)), mode='constant')[np.abs(shift):]

    assert scale <= 1.
    w = 1 + scale * (w - 1)
    return w


def transform_band_pass(D, scale: float, shift: float):
    """
    Applies a band-pass filter with a Hanning shape
    :param D: input spectrogram, shape = (num_mels, frames)
    :param scale: Difference between peak pass amplitude and stop amplitude.
    :param shift: Determines the center of the band. -0.5 places the center at the left edge, 0.5 at the right edge.
    :return: Filtered spectrogram
    """
    w = _get_1d_hann_filter(D.shape[0], scale=scale, shift=shift * D.shape[0])
    w_ex = np.repeat(w.reshape(w.shape[0], 1), D.shape[1], axis=1)
    D_ = np.multiply(w_ex, D)
    return D_


def transform_noise(D, snr: float = 1.):
    """
    Applies random noise with amplitude signal to noise ratio snr
    :param D: input spectrogram, shape = (num_mels, frames)
    :param snr: Signal to noise ratio
    :return: Noised spectrogram
    """
    row, col = D.shape
    gaussian = np.random.random((row, col))
    noise = 1 / (1 + snr)
    signal = 1 - noise
    D_ = np.add(noise * gaussian, signal * D)
    return D_


def transform_compress(D, thresh: float = 0.4, ratio: float = 5.):
    """
    Applies hard compression on the spectrogram (equivalent to hard knee, 0 attack and 0 release compression on a time domain signal)
    :param D: input spectrogram, shape = (num_mels, frames)
    :param thresh: set threshold as a factor of the max value in the spectrogram
    :param ratio: compression ratio
    :return: compressed spectrogram
    """
    thresh = thresh * np.max(D)
    compressed_pts = np.multiply(D, D > thresh)
    compressed_pts = compressed_pts * (1 / ratio) + thresh
    D_ = np.multiply(D, D <= thresh) + np.multiply(compressed_pts, D > thresh)
    D_ = (D_ / np.max(D_)) * np.max(D)
    return D_


def transform_random_roll_time(D):
    length = D.shape[1]
    shift = np.random.randint(0, length)
    return np.roll(D, shift, axis=1)


def transform_flip_time(D):
    return D[:, ::-1]


def transform_flip_freq(D):
    return D[::-1, :]


def get_spec_augmenter(amount='heavy', compress=True, filter=True, noise=True, roll=True):
    transforms = []
    if roll:
        transforms.append(SpecTransform(transform_random_roll_time))

    if amount == 'heavy':
        if compress:
            transforms.append(SpecTransform(transform_compress, thresh=(0.3, 1.), ratio=(1., 10.)))
        if filter:
            transforms.append(SpecTransform(transform_band_pass, scale=(0.8, 1.), shift=(-0.7, 0.7)))
        if noise:
            transforms.append(SpecTransform(transform_noise, snr=(1., 3.)))

    elif amount == 'medium':
        if compress:
            transforms.append(SpecTransform(transform_compress, thresh=(0.4, 1.), ratio=(1., 6.)))
        if filter:
            transforms.append(SpecTransform(transform_band_pass, scale=(0.5, 1.), shift=(-0.5, 0.5)))
        if noise:
            transforms.append(SpecTransform(transform_noise, snr=(1., 5.)))

    elif amount == 'light':
        if compress:
            transforms.append(SpecTransform(transform_compress, thresh=(0.5, 1.), ratio=(1., 4.)))
        if filter:
            transforms.append(SpecTransform(transform_band_pass, scale=(0.2, 1.), shift=(-0.4, 0.4)))
        if noise:
            transforms.append(SpecTransform(transform_noise, snr=(2., 8.)))

    if len(transforms) == 0:
        spec_transformations = None
    else:
        spec_transformations = SpecAugment(transforms)

    logger.info(f"Spec augment -- {amount} -- {[str(t) for t in transforms]}")

    return spec_transformations

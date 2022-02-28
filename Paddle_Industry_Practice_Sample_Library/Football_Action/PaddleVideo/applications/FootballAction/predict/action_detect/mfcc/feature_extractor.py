"""
audio feature extract
"""
# coding: utf-8
import os
import numpy as np
import pickle
import mfcc.vgg_params as vgg_params


def frame(data, window_length, hop_length):
    """
    frame
    """
    num_samples = data.shape[0]
    #print("window_length , hop_length", window_length, hop_length)
    #print("num_sample = ", num_samples)
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    #print(" num_frames = ", num_frames)
    shape = (num_frames, window_length) + data.shape[1:]
    #print(" shape = ", shape)
    strides = (data.strides[0] * hop_length, ) + data.strides
    #print("data.strides = ", data.strides)
    #print("strides = ", strides)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length):
    """
    periodic_hann
    """
    return 0.5 - (0.5 *
                  np.cos(2 * np.pi / window_length * np.arange(window_length)))


def stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    """
    stft_magnitude
    """
    frames = frame(signal, window_length, hop_length)
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
    """
    hertz_to_mel
    """
    return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz /
                                                 _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):
    """
    spectrogram_to_mel_matrix
    """
    nyquist_hertz = audio_sample_rate / 2.
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                         (lower_edge_hertz, upper_edge_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz,
                                         num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                                 hertz_to_mel(upper_edge_hertz),
                                 num_mel_bins + 2)
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                       (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                       (upper_edge_mel - center_mel))
        mel_weights_matrix[:,
                           i] = np.maximum(0.0,
                                           np.minimum(lower_slope, upper_slope))
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def log_mel_spectrogram(data,
                        audio_sample_rate=8000,
                        log_offset=0.0,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        **kwargs):
    """
    log_mel_spectrogram
    """
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    #print("audio_sample_rate = ", audio_sample_rate)
    #print("window_length_secs = ", window_length_secs)
    #print("window_length_sample ", window_length_samples)
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    #print("hop_length_samples ", hop_length_samples)
    fft_length = 2**int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    #print(" fft_lengt = ", fft_length)
    spectrogram = stft_magnitude(data,
                                 fft_length=fft_length,
                                 hop_length=hop_length_samples,
                                 window_length=window_length_samples)
    #print(" spectrogram.shape = ", spectrogram.shape)
    mel_spectrogram = np.dot(
        spectrogram,
        spectrogram_to_mel_matrix(num_spectrogram_bins=spectrogram.shape[1],
                                  audio_sample_rate=audio_sample_rate,
                                  **kwargs))

    return np.log(mel_spectrogram + log_offset)


def wav_to_example(wav_data, sample_rate):
    """
    wav_to_example
    """
    #sample_rate, wav_data = wavfile.read(wav_file)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    #wav_data = wav_data[:16000*30]
    #print(" wav_data ", wav_data.shape)
    #print(" wav_data ", wav_data.shape)
    pad_zero_num = int(sample_rate * (vgg_params.STFT_WINDOW_LENGTH_SECONDS -
                                      vgg_params.STFT_HOP_LENGTH_SECONDS))
    wav_data_extend = np.hstack((wav_data, np.zeros(pad_zero_num)))
    wav_data = wav_data_extend
    #print(" wav_data ", wav_data.shape)
    wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    #print(" wav_data after convert to -1 1", wav_data)
    #if wav_data.shape[0] > max_second * sample_rate:
    #    wav_data = wav_data[:max_second * sample_rate, :]
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
    #print(" wav_data after mean", wav_data.shape, len(wav_data.shape), wav_data)
    # Resample to the rate assumed by vgg.
    #if sample_rate != vgg_params.SAMPLE_RATE:
    #    wav_data = resampy.resample(wav_data, sample_rate, vgg_params.SAMPLE_RATE)
    log_mel = log_mel_spectrogram(
        wav_data,
        audio_sample_rate=vgg_params.SAMPLE_RATE,
        log_offset=vgg_params.LOG_OFFSET,
        window_length_secs=vgg_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vgg_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vgg_params.NUM_MEL_BINS,
        lower_edge_hertz=vgg_params.MEL_MIN_HZ,
        upper_edge_hertz=vgg_params.MEL_MAX_HZ)
    # Frame features into examples.
    features_sample_rate = 1.0 / vgg_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(
        round(vgg_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))

    example_hop_length = int(
        round(vgg_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = frame(log_mel,
                             window_length=example_window_length,
                             hop_length=example_hop_length)
    return log_mel_examples


def extract_pcm(pcm_file, sample_rate):
    with open(pcm_file, "rb") as f:
        pcm_data = f.read()
    audio_data = np.fromstring(pcm_data, dtype=np.int16)
    examples = wav_to_example(audio_data, sample_rate)
    return examples


if __name__ == "__main__":
    wav_file = sys.argv[1]
    print("wav_file = ", wav_file)
    with open(wav_file, "rb") as f:
        pcm_data = f.read()
    audio_data = np.fromstring(pcm_data, dtype = np.int16)
    examples_batch = wav_to_example(audio_data, 16000)
    print("examples_batch.shape", examples_batch.shape)   

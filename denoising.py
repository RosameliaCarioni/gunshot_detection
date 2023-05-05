from scipy.signal import butter, filtfilt, lfilter
import numpy as np
import scipy.io.wavfile as wavfile
import noisereduce as nr
import tensorflow as tf

# This function saves the denoised clips 
def save_denoised(reduced_noise, rate, destination_file):
    # because the denoised clips will be used by tf.audio.decode_wav and this only takes 16-bit files, the denoised audios are saved as int16
    # https://stackoverflow.com/questions/64813162/read-wav-file-with-tf-audio-decode-wav
    wavfile.write(destination_file, rate, reduced_noise.astype(np.int16)) 


# SPECTRAL GATING METHODS
def denoise_spectral_gating(file_name):
    rate, data = wavfile.read(file_name)
    data = data - data.mean() #center data  #TODO: maybe remove from here 
    reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=True)
    return reduced_noise, rate

def spectral(file_name, destination_file):
    reduced_noise, rate = denoise_spectral_gating(file_name)
    save_denoised(reduced_noise, rate, destination_file)



# LOW PASS FILTER METHODS 
def butter_lowpass_filter(wave, cutoff, sample_rate, order=4):
    b, a = butter(order, cutoff, fs=sample_rate, btype='low', analog=False)
    filtered_data = lfilter(b, a, wave)
    return filtered_data

def low_pass(file_name, cutoff, order, destination_file): 
    sample_rate, wave = wavfile.read(file_name)
    wave = wave - wave.mean() #center data  #TODO: maybe remove from here 
    denoised = butter_lowpass_filter(wave, cutoff, sample_rate, order)
    save_denoised(denoised, sample_rate, destination_file)


# BAND PASS FILTER METHODS
# https://dsp.stackexchange.com/questions/56604/bandpass-filter-for-audio-wav-file
# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

def butter_bandpass(lowcut, highcut, fs, filter_order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(filter_order, [low, high], btype='band', analog=False)
    return b, a

def apply_bandpass_filter(data, lowcut_freq, highcut_freq, sample_rate, filter_order):
    b, a = butter_bandpass(lowcut_freq, highcut_freq, sample_rate, filter_order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def band_pass(file_name, lowcut_freq, highcut_freq, destination_file, filter_order =4):
    # the cutoff freqs are in Hz
    sample_rate, wave = wavfile.read(file_name)
    wave = wave - wave.mean() #center data  #TODO: maybe remove from here 
    denoised = apply_bandpass_filter(wave, lowcut_freq, highcut_freq, sample_rate, filter_order)
    save_denoised(denoised, sample_rate, destination_file)

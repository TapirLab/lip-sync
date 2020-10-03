# -*- coding: utf-8 -*-
"""Python implementation of Tapir Lab.'s Acoustic Lip Synchronization.

This program extracts required statistics for training and matching phases
Functions are divided into two sub-groups as auxiliary and main functions

Main Functions
--------------
    Voice Activity Detection:
        VAD(1D_array,
            sampling_frequency=44100,
            average_phoneme_duration=0.06,
            noise_figure="constant",
            show_graph="False",
            )
            1D_array is the recording of voice.
            If it is required to see detected activity, make show_graph "True".
    Linear Predictive Coding Coefficients:
        lpc(correlation_coefficients, order_of_filter)
            correlation coefficients should be normalized.
    Parameter Extraction:
        parameter_extraction(1D_array, sampling_frequency, subsegments)
            1D_array is the recording of voice.
            Subsegments are constructed in VAD.

    Rest of the functions are called inside these three main functions.

Dependencies
------------
os
soundfile
numpy
scipy
matplotlib
math
------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Lip-Sync
%% -------------------
%% $Author: Halil Said Cankurtaran$,
%% $Date: October 2nd, 2020$,
%% $Revision: 1.0$
%% Tapir Lab.
%% Copyright: Halil Said Cankurtaran, Tapir Lab.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# import argparse
# import os
import soundfile

import numpy as np
import scipy.io as sio
import scipy.signal as signal  # lfilter, firwin
import matplotlib.pyplot as plt

from math import ceil, floor, log

# %% Auxiliary functions


def load_sound(filename):
    """Load recorded .wav file.

    Requires `soundfile` package to read record.

    Parameters
    ----------
    filename : str

    Returns
    -------
    x_t : numpy.ndarray, dtype('float64')
        One-dimensional array of sound recording
    sampling_frequency : int
        Sampling frequency of recording which is always 44100 Hz in study

    Raises
    ------
    FileNotFoundError
        If can not load the sound file.
    """
    try:
        x_t, sampling_frequency = soundfile.read(filename)

        return x_t, sampling_frequency

    except Exception as e:
        raise e


def load_mat_file(filename):
    """Use to load previously recorded .mat files.

    Requires `scipy.io` package to read record. `scipy.io` imported as sio
    Actually, ``scipy.io.loadmat(filename)`` loads the recorded file
    Additionally, read file parsed into previously named variables.

    Parameters
    ----------
    filename : str
        Exact path to .mat file

    Returns
    -------
    x_t : numpy.ndarray, dtype('float64')
        One-dimensional array of sound recording
    sampling_frequency : int
        Sampling frequency of recording which is always 44100 Hz in study
    number_of_audio_channels : int
        mono or stereo
    number_of_bits : int
        The number of bits to represent each sample in memory

    Raises
    ------
    FileNotFoundError
        If can not load the mat file.
    """
    try:
        mat_file = sio.loadmat(filename)  # loads .mat file

        # Parse loaded file
        x_t = mat_file["r_t_recorded"]
        sampling_frequency = mat_file["sampling_frequency"][0][0]  # save int
        number_of_audio_channels = mat_file["number_of_audio_channels"]
        number_of_bits = mat_file["number_of_bits"]

        return [x_t,
                sampling_frequency,
                number_of_audio_channels,
                number_of_bits,
                ]

    except Exception as e:
        raise e


def hist_bin_mid(array, neyman_pearson_bin_size=3):
    # Numpy returns start and end points of bins,
    # Matlab returns mid point of bins,
    # This function calculates mid points of histogram bins
    number_of_bins = floor(len(array)/neyman_pearson_bin_size)
    tmp, edges = np.histogram(array, number_of_bins)

    # Calculate center of the bins
    bin_mid = [(edges[i] + edges[i+1])/2 for i in range(0, len(edges) - 1)]

    return tmp, bin_mid


def normalize(r_t):
    # Normalizes given signal into [1,-1] interval.
    return r_t/max(abs(r_t))


def nextPow2(value):
    # Implementation of builtin matlab function
    # finds next power of two for given value
    return ceil(log(abs(value), 2))


def removeDC(r_t):
    # Removes the avereage (DC) from the signal
    return r_t - r_t.mean()


def lpFilter(x_t, sampling_frequency, cutoff, ntaps):
    """Filter given (x_t) signal with specified FIR low-pass filet.

    Requires `scipy.signal` package. `scipy.signal` imported as signal

    Parameters
    ----------
    x_t : numpy.ndarray, dtype('float64')
        One-dimensional array of sound recording
    sampling_frequency : int
        Sampling frequency of recording which is always 44100 Hz in study
    cutoff : int
        Cutoff frequency of filter
    ntaps : int
        Number of taps

    Returns
    -------
    filtered_signal : numpy.ndarray, dtype('float64')
    """
    nyq_rate = sampling_frequency / 2
    fir_coeff = signal.firwin(ntaps, cutoff/nyq_rate)  # create filter coeffs

    # filter x_t with all-zero low-pass filter
    filtered_signal = signal.lfilter(fir_coeff, 1.0, x_t)

    return filtered_signal


def reshape_signal(x_t, chunk_size):
    # To process signal column-wise, result is transposed
    number_of_chunks = floor(len(x_t)/chunk_size)  # calculate number of chunks
    # Remove residual part which may occur at the end of the signal.
    x_t_resized = x_t[:chunk_size*number_of_chunks]
    # Reshape signal to column vectors
    xt_reshaped = x_t_resized.reshape(number_of_chunks, chunk_size).transpose()

    return xt_reshaped


def avgEnergy(x_t_reshaped, scale="log"):
    """Calculate average energy of reshaped signal.

    Signal is reshaped according to following chunk_size calculation
    ``energy_bin_duration = 15e-3``
    ``chunk_size = floor(energy_bin_duration * sampling_frequency)``

    Energy bin duration

    Parameters
    ----------
    x_t_reshaped : numpy.ndarray, dtype('float64')
        This array is divided into equal chunks by given calculation
    scale : str, optional
        Logarithmic or linear calculation of energy of chunks
        It is logarithmic by default

    Returns
    -------
    log_energy/linear_energy : numpy.ndarray, dtype('float64')
        calculated energy of chunks
    """
    if scale.lower() == "log":
        log_energy = 10*np.log10((abs(x_t_reshaped)**2).mean(axis=0))
        return log_energy
    elif scale.lower() == "linear":
        linear_energy = (abs(x_t_reshaped)**2).mean(axis=0)
        return linear_energy
    else:
        print("please set scale to \"log\" or \"linear\" "
              "| default -> \"log\"")
        return 0


def optimumThreshold(r_t, option):
    """Calculate optimum noise figure in the signal to detect activity.

    Parameters
    ----------
    r_t : numpy.ndarray, dtype('float64')
        Signal to detect noise figure
    option : str
        This parameter must be given as adaptive/constant
        If constant, function returns -12. This is an emprical value

    Returns
    -------
    noise_figure : int
    """
    avg_bin_energy_in_dB = avgEnergy(r_t, "log")
    avg_bin_energy = avgEnergy(r_t, "linear")
    ndts = 3  # noise distribution three sigma
    tmp, bin_mid = hist_bin_mid(avg_bin_energy)  # returns mid points of bins

    if option == "adaptive":
        noise_figure_in_dB = 10 * log(bin_mid[ndts], 10)
    elif option == "constant":
        noise_figure_in_dB = -12
    else:
        print("please specify option as 'adaptive' or 'constant'")

    noise_figure = (max(avg_bin_energy_in_dB) + noise_figure_in_dB)

    return noise_figure


def autocorrbyfft(x_t, order=-2, mod="normalized"):
    """Calculate autocorraletaion via multiplication in frequency domain.

    Autocorrelation array will be sample averaged and normalized by default.
    However, if normalization is not required, a None type can be given.

    Parameters
    ----------
    x_t : numpy.ndarray, dtype('float64')
        Signal to autocorrelate
    order : integer, optional
        Order is required to provide a adequate but shorter array
        If order is not specified, function returns complete array
    mod : str, optional
        A normalized autocorrelation array is needed in LPC calculation
        If dont need to normalize, pass None as parameter

    Returns
    -------
    R_t : numpy.ndarray, dtype('float64')
        Autocorralation array of signal.
        If order is specified len(R_t) == order+1!
    """
    length_of_signal = len(x_t)  # get length of signal to calculate fft size
    fft_size = nextPow2(2*length_of_signal-1)  # calculate fft size

    # Transform signal to frequency domain
    X_t = np.fft.fft(x_t, 2**fft_size)
    # Calculate autocorrelation and transform signal to time domain
    # Signal is real, thus keep only real part
    R = np.real(np.fft.ifft(abs(X_t)**2))
    R = R / len(x_t)  # Sample average of autocorralation array

    if mod == "normalized":  # If normalized. It is default
        R = R / R[0]  # Normalize

    return R[:order+1]


def activityMask(x_t, chunk_size, noise_figure):
    """Detect activity in the given signal which is divided into chunks.

    Parameters
    ----------
    x_t : numpy.ndarray, dtype('float64')
        Signal to detect activity

    Returns
    -------
    activity_mask_flattened : numpy.ndarray, dtype('np.int8')
        An array of ones and zeros which indicates the activity in the signal
    """
    # Calculate average energy of bins
    avg_bin_energy_in_dB = avgEnergy(x_t, scale="log")
    # Calculate optimum threshold to detect activity
    if noise_figure == "constant":
        opt_threshold = optimumThreshold(x_t, "constant")
    if noise_figure == "adaptive":
        opt_threshold = optimumThreshold(x_t, "adaptive")
    # Decide activity in chunks by comparing enegry level to optimum threshold
    # Use comprehension to process in single line
    mask = [i > opt_threshold for i in avg_bin_energy_in_dB]
    # Each boolean in mask represents chunk_size of samples
    # Thus following line expands decision of activity to chunk_size of samples
    activity_mask = np.array([[i]*chunk_size for i in mask])
    # Convert boolean mask to binary array which has same size with x_t
    activity_mask_flattened = (activity_mask.flatten())*1

    return activity_mask_flattened


def construct_segments(activity_mask_flattened):
    """Find first and last samples of segments acc. to activity mask.

    Parameters
    ----------
        activity_mask_flattened : numpy.ndarray, dtype("np.int8")
            array of 1s and 0s which shows where the activity is

    Returns
    -------
    segments : numpy.ndarray, dtype("int64")
        An array of beginning and end points of segments
        Each row represents a particular segment of activity
    """
    # Find indices of first and last samples in a segment
    segment_start_idx = np.asarray(np.diff(activity_mask_flattened) > 0).nonzero()[0] + 1
    segment_stop_idx = np.asarray(np.diff(activity_mask_flattened) < 0).nonzero()[0]

    # If activity does not end before the last sample of recording
    if len(segment_start_idx) > len(segment_stop_idx):
        segment_stop_idx = np.insert(segment_stop_idx,
                                     len(segment_stop_idx),
                                     len(activity_mask_flattened))
    # If activity starts with the first sample there might be a missing start
    if len(segment_stop_idx) > len(segment_start_idx):
        segment_start_idx = np.insert(segment_start_idx, 0, 0)
    # There might be a third case where activity starts before first sample
    # and ends after last sample, in that case, lengths would be equal
    # however there segment start and finish samples will be mis-detected

    segments = np.array([segment_start_idx, segment_stop_idx]).transpose()

    return segments


def construct_subsegments(segments, sampling_frequency,
                          average_phoneme_duration):
    """Create an array of indices which represents the subsegments in segments.

    Parameters
    ----------
    segments : numpy.ndarray, dtype("int64")
        An array of beginning and end points of segments
    sampling_frequency : int
        Sampling frequency of recorded signal
    average_phoneme_duration : float, optional
        The length of average phoneme duration in seconds.

    Returns
    -------
    subsegments : numpy.ndarray, dtype("np.int8")
        An array of beginning and end points of subsegments
        subsegment[0] -> segment number
        subsegment[1] -> subsegment number
        subsegment[2] -> start of subsegment
        subsegment[3] -> end of subsegment
    """
    samples_per_subsegment = floor(average_phoneme_duration*sampling_frequency)
    sps = samples_per_subsegment
    number_of_segments = len(segments)
    count_of_subsegments = []

    if number_of_segments < 1:
        raise 'No segmentation found!'

    for k in range(0, number_of_segments):
        segment_length_in_samples = segments[k, 1] - segments[k, 0] + 1
        number_of_subsegments = 1
        if segment_length_in_samples >= sps:
            number_of_subsegments = floor(segment_length_in_samples / sps)
        count_of_subsegments.append(number_of_subsegments)

    subsegments = []
    segment_id = 0
    for k in count_of_subsegments:
        for i in range(0, k):
            subsegments.append([
                                segment_id,
                                i,
                                segments[segment_id, 0] + i*sps,
                                segments[segment_id, 0] + (i+1)*sps,
                                ])
        segment_id += 1

    return subsegments


def calculate_ffreqs(lpc_coeff, sampling_frequency):
    """Calculate formant frequencies.

    Finds roots of all-pole filter and converts to frequencies

    Parameters
    ----------
    lpc_coeff : numpy.ndarray, dtype('float64')
        An array of linear predictive coding coefficients
    sampling_frequency : int
        Sampling frequency of recorded signal

    Returns
    -------
    ffreqs : numpy.ndarray, dtype('float64')
        Formant frequencies
    """
    # Formants are the roots of all-pole transfer function
    # Following calculates roots of lpc coefficients
    # Calculate roots
    roots = np.roots(lpc_coeff)  # Calculate poles of voice model
    rts = roots[np.imag(roots) >= 0.01]  # Select positives of conjugate roots

    # Convert formant frequencies
    ffreqs = sorted(np.arctan2(np.imag(rts), np.real(rts))*sampling_frequency/(2*np.pi))

    return ffreqs

#%% Main functions


def VAD(x_t, sampling_frequency, avg_phn_dr, noise_fig, show_graph=False):
    """Detect voice activity and return segments, and subsegments.

    Parameters
    ----------
    x_t : numpy.ndarray, dtype('float64')
        One-dimensional array of sound recording
    sampling_frequency : int
        Sampling frequency of recorded signal
    avg_phn_dr : float
        average phoneme duration
    noise_fig : string
        "adaptive" or "constant" to decide optimum activity threshold
    show_graph = bool
        A parameter to show graph of activity detection if needed.

    Returns
    -------
    xt_processed_flattened : numpy.ndarray, dtype('float64')
        Loaded sound is processed before next steps
        This function returns normalized and mean extracted version of sound
    segments : numpy.ndarray, dtype("int64")
        An array of beginning and end points of voice activities
        Each row represents a particular part of the activity
    subsegments : numpy.ndarray, dtype("np.int8")
        An array of beginning and end points of subsegments
        subsegment[0] -> segment number
        subsegment[1] -> subsegment number
        subsegment[2] -> start of subsegment
        subsegment[3] -> end of subsegment
    """
    # Required variables to process x_t
    energy_bin_duration = 15e-3
    chunk_size = floor(energy_bin_duration * sampling_frequency)

    # Process x_t
    xt_normalized = normalize(x_t)
    xt_normalized_zero_mean = removeDC(xt_normalized)
    xt_processed_reshaped = reshape_signal(xt_normalized_zero_mean, chunk_size)

    # Create activity mask
    activity_mask_flattened = activityMask(xt_processed_reshaped,
                                           chunk_size,
                                           noise_fig)
    xt_processed_flattened = xt_processed_reshaped.transpose().flatten()

    if show_graph:
        plt.plot(xt_processed_flattened)
        plt.plot(activity_mask_flattened)

    segments = construct_segments(activity_mask_flattened)
    subsegments = construct_subsegments(segments,
                                        sampling_frequency,
                                        avg_phn_dr)

    return [xt_processed_flattened, segments, subsegments]


def lpc(r, n):
    """Calculate Linear Predictive Coefficients with autocorrelation method.

    Parameters
    ----------
    r : numpy.ndarray, dtype('float64')
        Autocorralation array of signal
    n : int
        Filter order

    Returns
    -------
    lpc_coeff : np.ndarray, dtype('float64')
        Linear predictive coding coefficients
    e : np.ndarray, dtype('float64')
        Estimation error array
    """
    order = n + 1

    # Construct required matrix
    a = np.zeros([order, order])
    k = np.zeros(order)
    e = np.zeros(order)

    # initilaziation of calculation
    a[:, 0] = 1.0
    k[1] = r[1]/r[0]
    a[1, 1] = k[1]
    e[1] = (1.0 - k[1]**2)*r[0]

    # Calculation of lpc coefficients
    for j in range(2, order):
        k[j] = (r[j] - sum([a[j-1, i]*r[j-i] for i in range(1, j)]))/e[j-1]
        a[j, j] = k[j]
        for i in range(1, j):
            a[j, i] = a[j-1, i] - k[j]*a[j-1][j-i]
        e[j] = (1-k[j]**2)*e[j-1]

    # Return only LPC Coefficients with correct order
    lpc_coeff = np.ones(order)
    # all-pole filter 1 - sum(lpc_coeff*z^(k)) coeffs are multiplied by -1
    # Get the results of correct order, first coefficient is set to +1
    lpc_coeff[1:] = -a[n::][0, 1:]

    return [lpc_coeff, e]


def parameter_extraction(x_t, sampling_frequency, subsegments):
    """Calculate formant freqs, mean, std. dev. and normalization vector.

    Parameters
    ----------
    x_t : numpy.ndarray, dtype('float64')
        One-dimensional array of sound recording
    sampling_frequency : int
        Sampling frequency of recorded signal
    subsegments : numpy.ndarray, dtype("np.int8")
        Indicies of subsegments with segment number and subsegment number.
        subsegment[0] -> segment number
        subsegment[1] -> subsegment number
        subsegment[2] -> start of subsegment
        subsegment[3] -> end of subsegment

    Returns
    -------
    formants : numpy.ndarray, dtype('float64')
        Formant frequencies of subsegments
        formants[0] -> segment number
        formants[1] -> subsegment number
    mean_lpc_freqs : numpy.ndarray, dtype('float64')
        Sample mean of formant frequencies
    vowel_formant_deviation_error_normalization_vector :
        Weightenin coeffs which will be used in mean square error estimation
    """
    # Reqired variables
    number_of_ffreqs = 3  # number of formant frequencies will be used
    order = floor(2 + sampling_frequency/1000)
    counter = 0
    # Create empty matrices to store data
    formants = np.zeros([len(subsegments), number_of_ffreqs + 2], np.float64)
    # lpc_coeffs = np.zeros([len(subsegments), order+1])

    # For each subsegment, find formant frequencies
    for subsegment in subsegments:
        subsegment_tmp = x_t[subsegment[-2]:subsegment[-1]]  # Slice of record
        r = autocorrbyfft(subsegment_tmp, order, "normalized")  # ACorr. Arr.
        lpc_coeff, e = lpc(r, order)  # Calc. of LPC coefficients of subsegment
        fftmp = calculate_ffreqs(lpc_coeff, sampling_frequency)  # root -> freq
        # Segment and subsegment numbers are saved as first two elements
        formants[counter, 0:2] = subsegment[0], subsegment[1]
        # Only reqiured part of the formant frequencies are returned
        formants[counter, 2:] = fftmp[:number_of_ffreqs]
        # lpc_coeffs[counter,:] = lpc_coeff # store coeffs of each subsegment
        counter += 1

    mean_lpc_freqs = np.mean(formants[:, -3:], 0)  # mean of ffreqs
    # ddof=1 to calculate sample mean -> 1/(N-1)
    std_lpc_freqs = np.std(formants[:, -3:], 0, ddof=1)  # std. dev. of ffreqs
    # Calculation of proposed weightening coefficients
    tmp = min(std_lpc_freqs) / std_lpc_freqs
    tmp = tmp/np.linalg.norm(tmp)
    vowel_formant_deviation_error_normalization_vector = tmp

    return [formants, mean_lpc_freqs,
            vowel_formant_deviation_error_normalization_vector]

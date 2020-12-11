# -*- coding: utf-8 -*-
"""Python implementation of Tapir Lab.'s Acoustic Lip Synchronization.

This program matches the vowels in the given file with previously trained ones.
If there is no match, the report includes a large number that is specified as 20
initially.

Main Functions
--------------
    Compare and Match:
        compare_and_match(sound_file,
                          trained_vowels,
                          avg_phoneme_duration,
                          noise_figure,
                          fps):
            Sound_file is test data. Will be used to matching with trained vowels
            The rest of the parameters explained in the docstring of the function
    Load Parameters:
        load_parameters(vowel)
            the vowel is the name of the file which includes repetitions of a
            specified vowel to use in the match stage.
            It should contain the mean of fundamental frequencies and
            vowel_formant_deviation_error_normalization_vector which is
            explained in the article.

Dependencies
------------
paramater_extraction
numpy
os
------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Lip-Sync
%% -------------------
%% $Author: Halil Said Cankurtaran$,
%% $Date: December 11th, 2020$,
%% $Revision: 1.1$
%% Tapir Lab.
%% Copyright: Halil Said Cankurtaran, Tapir Lab.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import os
import numpy as np
import parameter_extraction as pe


def load_parameters(vowel):
    """Use to load previously recorded parameters.

    It is assumed that parameters are saved into the sample_output folder

    Parameters
    ----------
    vowel : str
        vowel to load e.g. 'a', 'e' etc.

    Returns
    -------
    mean_lpc_freqs : numpy.ndarray, dtype('float64')
        Sample mean of first three formant frequencies
    vowel_formant_deviation_error_normalization_vector :
        Weightenin coeffs which will be used in mean square error estimation

    Raises
    ------
    Exception
        If can not load the file.
    """
    rf = os.path.join('.', 'sample_output' + os.sep)
    file_name = "%s" % vowel
    extension = ".npy"

    try:
        [mean_lpc_freqs, normalization_vector] = np.load(rf +
                                                         file_name +
                                                         extension,
                                                         allow_pickle=True,
                                                         )

        return [mean_lpc_freqs, normalization_vector]

    except Exception as e:
        raise e


def compare_and_match(sound_file,
                      trained_vowels,
                      avg_phoneme_duration,
                      noise_figure,
                      FPS,
                      ):
    """Detect and report vowels in sound_file via trained_voices.

    Parameters
    ----------
    sound_file : str
        Name of the sound file in the sample_input folder
    trained_vowels : list of strings
        List of vowels in the sample_ouput directory
    avg_phoneme_duration : float
        Average phoneme duration in seconds. Empirically it is set to 0.025
    noise_figure : str
        This parameter must be given as 'adaptive' or 'constant'
        If constant, function returns -12. This is an empirical value
    FPS : int
        frames per seconds to decide frame number of vowel

    Returns
    -------
    report : numpy.ndarray, dtype('float64')
        explanation of each element:
            report[0] -> segment number
            report[1] -> subsegment number
            report[2] -> start of subsegment in seconds
            report[3] -> end of subsegment in seconds
            report[4] -> detected vowel in subsegment
            report[5] -> frame number in a video with specified fps
    length : float
        length of original sound file in seconds
    """
    # Load sound_file to detect vowels
    directory = os.path.join('.', 'sample_input' + os.sep)
    x_t, sampling_frequency = pe.load_sound(directory + sound_file)
    length = len(x_t)/sampling_frequency

    # Load pre-calculated statistics of trained vowels
    # what ever the given vowels are, they will be sorted
    # in report, each vowel will be represented as corresponding array index
    trained_vowel = {}
    for vowel in sorted(trained_vowels):
        trained_vowel[vowel] = load_parameters(vowel)

    # Process the signal and detect where the voice activity is
    x_t, segments, subsegments = pe.VAD(x_t,
                                        sampling_frequency,
                                        avg_phoneme_duration,
                                        noise_figure,
                                        False,
                                        )

    # Calculate formants
    all_parameters = pe.parameter_extraction(x_t,
                                             sampling_frequency,
                                             subsegments,
                                             )
    # Parse array to sub-elemets to get required formant information
    [formants, a, b] = all_parameters

    # Initiliaze parameters to match trained data and test data
    score_threshold = 120
    frame_rate = 1/FPS

    # Initilize required arrays and variables
    report = np.zeros([len(subsegments), 6])
    scores = np.zeros([len(formants), len(trained_vowels)])
    counter = 0

    # for each subsegment
    for elem in formants:
        # calculate socres for each subsegment
        tmp_scores = np.zeros(len(trained_vowels))
        i = 0
        for vowel in trained_vowel:
            mean_ff = trained_vowel[vowel][0]
            normalization_vector = trained_vowel[vowel][1]
            tmp_scores[i] = np.sqrt(np.mean(abs((mean_ff - elem[2:]) *
                                                normalization_vector)**2))
            i += 1

        scores[counter][:] = tmp_scores  # May be returned to analyze scores

        # Decide if there is any match
        min_score = tmp_scores.min()
        idx = np.where(tmp_scores == min_score)[0][0]
        if min_score < score_threshold:
            score_out = idx
        else:
            score_out = 20  # A large number to indicate no match

        # Construct elements of report
        subsegment_start_in_ms = subsegments[counter][2]/sampling_frequency
        subsegment_stop_in_ms = subsegments[counter][3]/sampling_frequency
        frame_idx = round(subsegment_start_in_ms/frame_rate)
        report[counter, :] = [elem[0],
                              elem[1],
                              subsegment_start_in_ms,
                              subsegment_stop_in_ms,
                              score_out,
                              frame_idx,
                              ]
        counter += 1

    return report, length

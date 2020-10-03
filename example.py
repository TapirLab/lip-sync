# -*- coding: utf-8 -*-
"""Python implementation of Tapir Lab.'s Acoustic Lip Synchronization.

This is an example to show how parameter_extraction and match are used.

Dependencies
------------
paramater_extraction
match
numpy
os
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

import os
import numpy as np
import parameter_extraction as pe
from match import compare_and_match

# Recordings of vowels, sound file to lip-synchronize, video background and
# mouth shapes are placed into the sample_input folder
sample_input = os.path.join("." + os.sep, "sample_input" + os.sep)
# Calculated statistics, report of matching phase, frame-shape list and
# lip-synchronized animation video will be saved to sample_output folder
sample_output = os.path.join("." + os.sep, "sample_output" + os.sep)

# Specify the wovels to be trained
vowels = ['a', 'e', 'i', 'o', 'u']
extension = ".wav"
# Specify the test sound file and report file name
test_sound = "test_sound.wav"
output_file = "report.txt"

# Calculate statistical parameters for each vowel
for vowel in vowels:
    # In sample inputs, each vowel is repeated at least five times
    # Also, hiss noise is removed from recordings to increase performance
    x_t, sampling_frequency = pe.load_sound(sample_input+vowel+extension)

    # Process the signal and detect where the voice activity is
    average_phoneme_duration = 0.06  # An experimental value
    # Optimum threshold is detected with -12dB constant noise figure,
    # In case of need, "adaptive" version can be used to calculate opt. thresh.
    noise_figure = "constant"
    # Detects where the activity is and shows the graph for visual inspection
    # show_graph can be set to False if graph is not required
    x_t, segments, subsegments = pe.VAD(x_t,
                                        sampling_frequency,
                                        average_phoneme_duration,
                                        noise_figure,
                                        show_graph=True,
                                        )

    # Calculate parameters of train data
    # To keep line short, initially all parameters are saved into parameters
    parameters = pe.parameter_extraction(x_t, sampling_frequency, subsegments)

    # Parse parameters variable to sub-variables
    [formants, mean_ffreqs, normalization_vector] = parameters

    # Formants will be used in the matching, no need to save for each now
    np.save(sample_output+vowel, [mean_ffreqs, normalization_vector])

# Match test data with trained vowels and save report as a txt file

average_phoneme_duration = 0.025  # An experimental value
# Optimum threshold is detected adaptively for test recording
noise_figure = "adaptive"
# FPS is specified to detect the frame number of mouth shape.
# Report file will include frame numbers and corresponding mouth shape
# However, reports.txt will be processed again in order to fix frame repetition
FPS = 10

# Voice acitivity detection and parameter extaction is performed in function
# For the sake of explanation they were performed 1by1 in parameter_extraction
# Since statistical parameters of each vowel is calculated and saved before,
# Only list of vowels are given as parameter.
# For the detailed structure of report.txt, docstring should be read.
report, length = compare_and_match(test_sound,
                                   vowels,
                                   average_phoneme_duration,
                                   noise_figure,
                                   FPS)

# in report a-> 0, e -> 1, i -> 2, o -> 3, u -> 4
np.savetxt(sample_output+output_file, report, header='%d' % length)

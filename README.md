# Tapir Lab.'s Lip-Sync Method
![Tapir Lab.](http://tapirlab.com/wp-content/uploads/2020/10/tapir_logo.png)
This is the official repository of the Tapir Lab.'s, “An Acoustic Signal Based Language Independent Lip Synchronization Method and Its Implementation via Extended LPC” paper which has been accepted for publication in 28th IEEE Conference on Signal Processing and Communications Applications, October 5-7, 2020, ~~Gaziantep~~ (Converted to Online due to the COVID-19 pandemic), Turkey.

## Description

This is the implementation of a language-independent lip-sync method that is based on extended linear predictive coding. The proposed method operates on the recording of acoustic signals which are acquired by a standard single-channel off–the–shelf microphone and exploits the statistical characteristics of acoustic signals produced by human speech. The mathematical background and detailed description of the method can be found in the paper.

Tapir Lab.'s Lip-Sync method consists of parameter extraction and matching phases. In order to calculate statistical parameters of vowels, one should pronounce each vowel several times and record them separately. In the parameter extraction phase, `VAD()` detects where the voice activity is, and then, `parameter_extraction()` calculates formant frequencies and their statistics like mean, standard deviation, and proposed _vowel formant deviation error normalization vector_. In the matching phase, voice activity in test recording is detected and formant frequencies of each active subsegment are calculated. After these calculations, lip-synchronization is performed by calculating the weighted mean square error between formant frequencies of each subsegment and previously trained vowels. The minimum of the errors is saved as the correct match if it is under the pre-determined threshold. If all the errors are above the threshold, the system denotes non-match with a special number.

 

## Prerequisites

* Python3
* ffmpeg
* Other necessary packages can be installed via `pip install -r requirements.txt`

**Note**: _requirements.txt_ includes opencv-python package. This should be noted that this package only includes fundamental functionalities of OpenCV library. Therefore, in case of need OpenCV should be installed separately.

## Folder Structure

```
Lip-Sync
|── sample_input
|   |── Train data, test data, mouth shapes, video background
|── sample_output
|   |── Train parameters, report, mouth shapes of frames, produced videos
|── example.py
|── match.py
|── parameter_extraction.py
|── report_to_anim.py
|── requirements.txt
|── LICENSE
```
## Example 

In order to ease the learning process of the method, a working example is provided in `example.py` and each step is explained below.

### Dependencies, Input and Output Files

```python
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
```
All environmental variables are initialized on the top of the file. In the next section, parameters will be calculated for each vowel in the `vowels` list and saved to the sample_output file.

### Parameter Extraction

```python
# Calculate statistical parameters for each vowel
for vowel in vowels:
    # In sample inputs, each vowel is repeated at least five times
    # Also, hiss noise is removed from recordings to increase performance
    x_t, sampling_frequency = pe.load_sound(sample_input+vowel+extension)

    # Process the signal and detect where the voice activity is
    average_phoneme_duration = 0.06  # An experimental value
    # Optimum threshold is detected with -12dB constant noise figure,
    # In case of need, the "adaptive" version can be used to calculate opt. thresh.
    noise_figure = "constant"
    # Detects where the activity is and shows the graph for visual inspection
    # show_graph can be set to False if graph is not required
    x_t, segments, subsegments = pe.VAD(x_t,
                                        sampling_frequency,
                                        average_phoneme_duration,
                                        noise_figure,
                                        show_graph=False,
                                        )

    # Calculate parameters of train data
    # To keep line short, initially all parameters are saved into parameters
    parameters = pe.parameter_extraction(x_t, sampling_frequency, subsegments)

    # Parse parameters variable to sub-variables
    [formants, mean_ffreqs, normalization_vector] = parameters

    # Formants will be used in the matching, no need to save for each now
    np.save(sample_output+vowel, [mean_ffreqs, normalization_vector])
```
In the second part of the example, means of formant frequencies and _vowel formant deviation error normalization vector_ are calculated and saved to the `sample_output` folder for each vowel. These statistical parameters will be used in the matching phase.

### Lip Synchronization

```python
# Match test data with trained vowels and save the report as a txt file

average_phoneme_duration = 0.025  # An experimental value
# Optimum threshold is detected adaptively for test recording
noise_figure = "adaptive"
# FPS is specified to detect the frame number of mouth shapes.
# Report file will include frame numbers and corresponding mouth shape
# However, reports.txt will be processed again in order to fix frame repetition
FPS = 10

# Voice activity detection and parameter extraction is performed in function
# For the sake of explanation they were performed 1by1 in parameter_extraction
# Since the statistical parameters of each vowel is calculated and saved before,
# Only a list of vowels is given as a parameter.
# For the detailed structure of report.txt, the docstring should be read.
report, length = compare_and_match(test_sound,
                                   vowels,
                                   average_phoneme_duration,
                                   noise_figure,
                                   FPS)

# in report a-> 0, e -> 1, i -> 2, o -> 3, u -> 4
np.savetxt(sample_output+output_file, report, header='%d' % length)
```
In the last part of the example, lip-synchronization is performed inside the `compare_and_match` function, and the report is saved into the `sample_output` folder. The animation phase needs only the `reports.txt` to create a video.

### Animation

In the animation phase, `reports.txt` should be processed before producing the video. Since FPS and average phoneme duration is different, there will be repetitions of frames with different mouth shapes in the report. In order to select the best match, repetitions of mouth shapes for each frame will be calculated and the largest one will be selected as the correct match. In case of equality in repetitions, the first corresponding mouth shape will be selected as the correct match. The mouth will be at rest when there is no detected voice activity. Also, if there is activity but no match with any of the vowels in the list, a random mouth shape will be returned.

Report to the animation is not generalized for each list of vowels due to a large number of possibilities in mouth shapes but it can be easily modified for different use cases. 

## License and Citation

The software is licensed under the MIT License. Please cite the following proceeding if you use this code:
```
@INPROCEEDINGS{tapir_2020_lipsync,
    author={Halil Said Cankurtaran and Ali Boyacı and Serhan Yarkan},
    title={{An Acoustic Signal Based Language Independent Lip Synchronization Method and Its Implementation via Extended LPC}},
    booktitle={Proc. of the 28th IEEE Conference on Signal Processing and Communications Applications},
    year={2020},
    month=oct # " 5--7, ",
    address={Online}}
```
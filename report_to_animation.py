"""Python implementation of Tapir Lab.'s Acoustic Lip Synchronization.

This program takes report as input and produces animation video

Dependencies
------------
os
opencv (cv2)
subprocess
numpy
pydub
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
import cv2
import subprocess
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks


def report_to_frames(path_to_report_file):
    """Process produced report file to decide corresponding frame mouth shape
    of each frame and saves to `sample_output` as `frame_shapes.txt`

    Parameters
    ----------
    path_to_report_file : string
        path to produced report file
    """

    sample_output = os.path.join("." + os.sep, "sample_output" + os.sep)
    output_file = "frame_shapes.txt"

    report = np.loadtxt(path_to_report_file)
    f = open(path_to_report_file)
    length = int(f.readline().split()[1])
    f.close()

    vowels = report[:, 4].astype(np.int64)
    frame_idx = report[:, 5].astype(np.int64)

    max_frame_idx = max(frame_idx)
    residual = 10*length - max_frame_idx
    if residual < 0:
        residual = 0

    number_of_frames = max_frame_idx + residual
    frames = np.bincount(frame_idx)
    frame_shapes = np.zeros([number_of_frames+1, 2], np.int64)
    frame, start = 0, 0
    for count in frames:
        if count == 0:
            frame_shapes[frame] = frame+1, 0
        else:
            number_of_vowels = np.bincount(vowels[start:start+count])
            max_count = np.count_nonzero(number_of_vowels == max(number_of_vowels))
            if max_count > 1:
                frame_shapes[frame] = frame+1, vowels[start]
            else:
                greatest = max(number_of_vowels)
                vowel_idx = np.where(number_of_vowels == greatest)[0][0]
                frame_shapes[frame] = frame+1, vowel_idx
        start = start + count
        frame = frame + 1

    for i in range(max_frame_idx, number_of_frames):
        frame_shapes[i] = i+1, 0

    np.savetxt(sample_output+output_file, frame_shapes)


def frame_number2img(frame_no, frames_dict, mouth_images):
    """Find corresponding image of frame based representative values if frame
    does not exist in dictionary, return "silence" mouth shape.
    
    Parameters
    ----------
    frame_no : int
        Corresponding frame number
    frames_dict : dictionary
        A dictionary of frames and corresponding representative values
    mouth_images :
        list of mouth images. The index corresponds to representative value
    
    Returns:
    -------
    mouth_images : numpy.ndarray
        image of mouth shape
    """
    if frame_no in frames_dict:
        i = frames_dict[frame_no]
        if i == 20:  # Not matched with ant vowels
            i = np.random.randint(5, 10)
        if i == 0:  # No voice activity
            i = 10  # Rest image
        return mouth_images[i]
    return mouth_images[-1]


if __name__ == "__main__":
    sample_input = os.path.join("." + os.sep, "sample_input" + os.sep)
    sample_output = os.path.join("." + os.sep, "sample_output" + os.sep)
    reportFileName = sample_output + "report.txt"
    audioFileName = sample_input + "test_sound.wav"
    backgroundImage = sample_input + "bg.png"
    videoFileName = sample_output + "output.mp4"
    finalVideoFile = sample_output + "final.mp4"
    FPS = 10

    report_to_frames(reportFileName)  # create frame_shapes.txt

    frames = np.loadtxt(sample_output+"frame_shapes.txt")  # Read frame_shapes
    frames_dict = {}  # Create an empt dict to store frame # and mouth shape
    for frame in frames:  # Fill frames dictionary
        frames_dict[int(frame[0])] = int(frame[1])

    # Read mouth images
    mouths = "mouths" + os.sep
    a_img = cv2.imread(sample_input + mouths + "AI.png")
    e_img = cv2.imread(sample_input + mouths + "E.png")
    o_img = cv2.imread(sample_input + mouths + "O.png")
    i_img = cv2.imread(sample_input + mouths + "AI.png")
    u_img = cv2.imread(sample_input + mouths + "U.png")
    x_img = cv2.imread(sample_input + mouths + "etc.png")
    l_img = cv2.imread(sample_input + mouths + "L.png")
    wq_img = cv2.imread(sample_input + mouths + "WQ.png")
    fv_img = cv2.imread(sample_input + mouths + "FV.png")
    mbp_img = cv2.imread(sample_input + mouths + "MBP.png")
    rest_img = cv2.imread(sample_input + mouths + "rest.png")
    mouth_images = [a_img, e_img, i_img, o_img, u_img,
                    x_img, fv_img, l_img, mbp_img, wq_img, rest_img]

    bg = cv2.imread(backgroundImage)  # read background image
    w, h = bg.shape[1], bg.shape[0]  # get dimensions of background image

    sound = AudioSegment.from_file(audioFileName)  # read audio

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # set up video format
    out = cv2.VideoWriter(videoFileName, fourcc, FPS, (w, h))

    # Divide sound file into chunks specified by FPS and write mouth shape
    frame = 1
    for chunk in make_chunks(sound, 1000/FPS):
        tmp = bg.copy()  # Copy background to a temporary image
        # Get correct mouth shape
        mouth_img = frame_number2img(frame, frames_dict, mouth_images)
        y_start, y_end = 0, mouth_img.shape[0]  # Coords to write
        x_start, x_end = w - mouth_img.shape[1], w  # Coords to write
        tmp[y_start:y_end, x_start:x_end] = mouth_img  # Write mouth on bg
        out.write(tmp)  # Write to video object
        frame = frame + 1

    # Release out object
    out.release()

    # Merge video with sound file
    subprocess.call(
                    ['ffmpeg',
                     '-i', videoFileName,
                     '-i', audioFileName,
                     '-shortest', '-c:v',
                     'copy', '-c:a',
                     'aac', '-b:a',
                     '256k', '-y',
                     finalVideoFile]
                    )

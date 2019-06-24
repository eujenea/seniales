import numpy as np
import librosa as lib
import scipy
import pathlib
import pyaudio
import wave
from dtw import dtw
import numpy as np
from numpy.linalg import norm
from numpy import fft
from scipy.io import wavfile

#ARCHIVO PARA TESTEO DE LA FUNCION MFCC

signal, fm = lib.load('derecha_11k/derecha_1.wav')

print("Vector: ", signal)


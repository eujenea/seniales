import librosa as lib
import librosa.display as disp
# import pyaudio
import matplotlib.pyplot as plt
from keyboard import is_pressed
from numpy.linalg import norm
import numpy as np
from numpy import fft
# import wave
import scipy
from utils import readMFCCFromFile
from utils import recordAudio
from utils import makeDtw
from dtw import dtw
# from peakutils.plot import plot as pplot
from matplotlib import pyplot
print("Iniciando")

# =====================================
# CARGA DE DATOS
# Code here:

file ="mfccFromAudio" #Nombre de archivo    
datos = readMFCCFromFile(file+".npy") #Leo el archivo.
print("Se cargaron los datos: ",datos.shape) #muestro los datos. datos es un numpyarray de dos posiciones, en la posicion 0, tiene los mfcc; en la 1, la etiqueta.


# ===========================================
# KEY WAITING LOOP

while True:  # making a loop
    try:  # used try so that if user pressed other than the given key error will not be shown
        if is_pressed('r'):  # if key 'q' is pressed 
            
            print('Iniciaste Grabacion')
            recordAudio()
            
            #dtw es una matriz que contiene elementos: [DistanciaDTW, IDPalabra]
            #Este vector debe enviarse al knn para poder decidir que palabra sera la seleccionada
            dtw = makeDtw(datos)
            print("Vector de distancias: ", dtw.shape)
        elif is_pressed("q"):
            print("Fin del programa")
            break
        else:
            pass
    except:
        break  # if user pressed a key other than the given key the loop will break
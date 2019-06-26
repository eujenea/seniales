import librosa as lib
import librosa.display as disp
import matplotlib.pyplot as plt
from keyboard import is_pressed
from numpy.linalg import norm
import numpy as np
from numpy import fft
import scipy
import utils as ut
from dtw import dtw
from compara import knnMFCC
import globals

print("Iniciando")

# =====================================
# CARGA DE DATOS
# Code here:


datos = ut.readMFCCFromFile(globals.FILE+".npy") #Leo el archivo.
print("Se cargaron los datos: ",datos.shape) #muestro los datos. datos es un numpyarray de dos posiciones, en la posicion 0, tiene los mfcc; en la 1, la etiqueta.


# ===========================================
# KEY WAITING LOOP

print("Presioná la tecla 'r' para grabar.")
while True:  # making a loop
    #try:  # used try so that if user pressed other than the given key error will not be shown
        if is_pressed('r'):  # if key 'q' is pressed 
            
            print('Iniciaste Grabacion')
            ut.recordAudio()
            knnMFCC(datos, ut.promediaMfccAudioGrabado())


            #dtw es una matriz que contiene elementos: [DistanciaDTW, IDPalabra]
            #Este vector debe enviarse al knn para poder decidir que palabra sera la seleccionada
            #dtw = makeDtw(datos)
            #print("Vector de distancias: ", dtw.shape)

            print("Presioná la tecla 'r' para grabar.")
        elif is_pressed("q"):
            print("Fin del programa")
            break
        else:
            pass
    #except:
    #    print("ha ocurrido un problema")
    #    break  # if user pressed a key other than the given key the loop will break
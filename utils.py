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

def makeDtw(datos):
    # =====================================
    # OBTENCION DE MFCC DEL NUEVO AUDIO
    NMFCC = 30

    y, sr = lib.load('./audio.wav')
    mfccAudio = lib.feature.mfcc(y, sr,n_mfcc=NMFCC)

    datosDtw = []

    # =====================================
    # CALCULO DE LAS DISTANCIAS DTW CON LA BD
    for i in range(datos.shape[0]):

        print("I: ", i)
        dist, cost, acc_cost, path = dtw(mfccAudio.T, datos[i,0].T, dist=lambda x, y: norm(x - y, ord=2))
        print("dtw: señal vs ", datos[i,1], " = ", dist)
        datosDtw.append([dist, datos[i,1]])

    datosDtw = np.array(datosDtw)
    return datosDtw
    # print("Vector de distancias: ", datosDtw.shape)

# Funcion para grabar 1 segundo de audio, y almacenarlo en un wav
def recordAudio():
    # =====================================
    # GRABACION DESDE MICROFONO
    # Code here:

    CHUNK = 2**11
    RATE = 44100

    p=pyaudio.PyAudio()
    stream=p.open(  format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frame = []
    print("se graba")
    for i in range(int(22*44100/(1024*24))): #go for a few seconds
        data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
        frame.append(data)
        #data
    print("se grabó")
    #lib.output.write_wav('./derecha_11k/audio.wav', cb, RATE)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    waveFile =wave.open('./audio.wav','wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frame))
    waveFile.close()



doc = """
guarda los coeficientes cepstrales en escala de mel de los audios.
Depende de la disposicion de carpetas.

parámetros:
frec: string -> indica la frecuencia de los audios (está en el nombre de las carpetas)
cantWAVs: int -> cantidad de audios por carpeta
NMFCC: int -> cantidad de coeficientes a calcular
file: string -> nombre de archivo que se quiere guardar (sin extension)
"""
def saveMFCCToFile(frec,cantWAVs,NMFCC,file):

    root='./'
    dale = "dale"
    izquierda = "izquierda"
    derecha = "derecha"
    para = "para"
    
    
    # PATHS: genera ./izquiera_11k
    dalePATH = pathlib.Path(root,dale+"_"+frec) 
    derechaPATH = pathlib.Path(root,derecha+"_"+frec)
    izquierdaPATH = pathlib.Path(root,izquierda+"_"+frec)
    paraPATH = pathlib.Path(root,para+"_"+frec)

    paths = [dalePATH, derechaPATH, izquierdaPATH, paraPATH]
    archivs = [dale, derecha, izquierda, para]

    y=[]
    for i in range(len(paths)):
        for num in range(1,cantWAVs):
            archivo = pathlib.Path(paths[i],archivs[i]+'_'+str(num)+'.wav')
            # print(archivo)
            # print(paths[i])
            # print(archivo)
            audio, frecMuestreo= lib.load(archivo)
            melCoef = lib.feature.mfcc(audio, frecMuestreo,n_mfcc=NMFCC)
            y.append([ melCoef,  i])
            # y.append([ melCoef,  archivs[i]])
            

    np.save(file,np.array(y))


    pass

def readMFCCFromFile(file):
    return np.load(file, allow_pickle=True)
    pass


if __name__ == "__main__":
    print()

    print("Si te da error, checá que este archivo esté en el mismo directorio donde están las carpetas con los audios.")
    print("Por ejemplo: en el directorio del proyecto tenés que tener.")
    print("./readData.py")
    print("./dale_11k/dale_1.wav")
    print("./dale_11k/dale_2.wav")
    print("./dale_11k/dale_3.wav")
    print("etc.")
    print()


    file ="mfccFromAudio" #Nombre de archivo
    saveMFCCToFile("11k",8,30,file) #Realizo el guardado 
    datos = readMFCCFromFile(file+".npy") #Leo el archivo.
    print(datos.shape) #muestro los datos. datos es un numpyarray de dos posiciones, en la posicion 0, tiene los mfcc; en la 1, la etiqueta.

    pass



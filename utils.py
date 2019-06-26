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
import globals
import math

def makeDtw(datos):
    # =====================================
    # OBTENCION DE MFCC DEL NUEVO AUDIO
    NMFCC = 30

    y, sr = lib.load('./audio.wav')
    #mfccAudio = lib.feature.mfcc(y, sr,n_mfcc=NMFCC)
    mfccAudio = mfcc(y, sr, globals.NMFCC, 25, 10)

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
            #melCoef = lib.feature.mfcc(audio, frecMuestreo,n_mfcc=NMFCC)
            melCoef = mfcc(audio, frecMuestreo, NMFCC)
            y.append([ melCoef,  i])
            # y.append([ melCoef,  archivs[i]])
            

    np.save(file,np.array(y))


    pass

def readMFCCFromFile(file):
    return np.load(file, allow_pickle=True)
    pass


#Lee el audio grabado del microfono, calculo los mfcc con nuestra funcion, promedia los valores de las ventanas y devuelve el resultado. Este se pasa derecho a KNN
def promediaMfccAudioGrabado():
    y, sr = lib.load('./audio.wav')
    #mfccAudio = lib.feature.mfcc(y, sr,n_mfcc=30)
    mfccAudio = mfcc(y, sr, globals.NMFCC, 25, 10)
    audioMfccProm = np.mean(mfccAudio.T, axis=0)
    return audioMfccProm


#==================================================
#CALCULO MFCC
#==================================================
def fmel(fhz):
    return 1000*np.log2(1+(fhz/1000))

def fmelinv(mel):
    return 1000*(2**(mel/1000)-1)

def makeFiltro(m, f,ksamples):
    y = np.zeros((ksamples))
    for k in range(ksamples):
        if(k<f[m-1]):
            y[k]=0
        elif(f[m-1]<=k and k<=f[m]):
            y[k]= (k-f[m-1])/(f[m]-f[m-1])
        elif (f[m]<=k and k<=f[m+1]):
            y[k]=(f[m+1]-k)/(f[m+1]-f[m])
        else:
            y[k]=0
    return y

#MFCC
#Parametros:
#    signal: señal de audio a calcular los coeficientes de mel
#    fm: frecuentcia a la que esa señal fue muestreada
#    windowLenght: tamaño de la ventana a usar (en milisegundos)
#    windowStep: indica cada cuanto se ventanea (en milisegundos)

def mfcc(signal, fm, cantCoef=30, windowLenght=25, windowStep=20):

    #VENTANEO--------------------------------------------------------
    samplesLength = math.floor((windowLenght*0.001)*fm)
    print("SamplesLen: ", samplesLength)

    samplesStep= math.floor((windowStep*0.001)*fm)
    print("SamplesStep: ", samplesStep)

    cantVentanas = math.ceil(signal.size/samplesStep)
    print("CantVentanas: ", cantVentanas)

    framedSignal = []

    for i in range(0, cantVentanas):

        #Tomo una porcion de la señal de longitud samplesLength
        frame = signal[i*samplesStep:i*samplesStep+samplesLength]
    
        #Si estamos en las ultimas ventana verifico que la ventana sea del tamaño samplesLength
        #De no ser asi agrego ceros al final
        if frame.shape[0] != samplesLength:
            #print("Ventana incompleta")
            frame = np.pad(frame,(0, samplesLength - frame.shape[0]), 'constant', constant_values=0)
            #print(frame)

        #Aplico Hamming a la ventana
        frame = np.hamming(samplesLength)*frame

        #Agrego ventana a la lista de ventanas
        framedSignal.append(frame)
            
    framedSignal = np.array(framedSignal)
    #print("framed signal shape: ", framedSignal.shape)

    fftSignal = []
    #CALCULO ENERGIA DE LA FFT (a cada ventana)
    for i in range(framedSignal.shape[0]):

        aux = framedSignal[i, :]
        auxfft = fft.fft(aux)
        espectro = (1/samplesLength)*(abs(auxfft)**2)

        #Debo quedarme con la primera mitad del espectrograma?
    
        fftSignal.append(espectro[: int(np.ceil(samplesLength/2))])

    #print("fftSignal Size ", len(fftSignal))
    #print("ventana fft ", fftSignal[0].shape[0])

    # BANCO DE FILTROS DE MEL
    # planteamos un minimo y un maximo en frecuencias, pasamos a mels, hacemos un equi espaciado entre esos valores en mel
    # convertimos los valores equiespaciados en Hz
    cantCoef
    nfft = fftSignal[0].shape[0] #la cantidad de samples de media ventana del espectro dde fourier
    min = 300 #hz
    max = fm/2 #hz

    #pasamos a mels los limites

    minMel = fmel(min)
    maxMel = fmel(max)

    melAxis = np.linspace(minMel, maxMel, cantCoef+2)
    herzAxis = np.array([ fmelinv(melAxis[i]) for i in range(melAxis.shape[0])])

    samplesAxis = np.array([np.floor((nfft)*herzAxis[i]/(fm/2)) for i in range(herzAxis.shape[0])])

    filtros = np.zeros((cantCoef, nfft))
    for m in range(cantCoef):
        filtros[m]=makeFiltro(m+1,samplesAxis,nfft)
        #plt.plot(filtros[m,:])

    #plt.show()

    melArray=[]
    for i in range(nfft):
        aux=np.dot(fftSignal[0],filtros.T)
        melArray.append(aux)

    melArray = np.array(melArray)
    return melArray.T

if __name__ == "__main__":


    file ="mfccFromAudio" #Nombre de archivo
    saveMFCCToFile("11k",8,globals.NMFCC,globals.FILE) #Realizo el guardado 
    datos = readMFCCFromFile(globals.FILE+".npy") #Leo el archivo.
    print(datos.shape) #muestro los datos. datos es un numpyarray de dos posiciones, en la posicion 0, tiene los mfcc; en la 1, la etiqueta.

    pass



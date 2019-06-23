import numpy as np
import librosa as lib
import scipy
import pathlib
from scipy.io import wavfile

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



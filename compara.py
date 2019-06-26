from utils import saveMFCCToFile as save
from utils import readMFCCFromFile as read
import utils as ut
import librosa as lib
import numpy as np
from sklearn import neighbors

#================================
# KNN BASADO EN CADA MFCC COMO UNA DIMENSION, DONDE SE BUSCA LA MENOR DISTANCIA ENTRE
# LAS CALCULADAS A PARTIR DE LA DISTANCIA EUCLIDEA TOMANDO TODAS LAS DIMENSIONES
#Parametros:
#    y: es la base de datos para pasar a knn -> numpy array con los mfcc de todos los audios
#    newcome: dato a etiquetar
#    devuelve la etiqueta como un entero de 0 a 3
def knnMFCC(y, newcome):

    data = y[:,0]
    etiquetas = y[:,1] #.flatten().tolist()

    dataProm=np.zeros((data.shape[0],data[0].shape[0]))

    for idx in range(len(data)):
        # Probar mediana
        #dataProm.append(np.mean(commando,axis=0).flatten().tolist())
        commando=data[idx]
    
        dataProm[idx,:] = np.mean(commando.T,axis=0)

    knn =  neighbors.KNeighborsClassifier(5)
    knn.fit(dataProm.tolist(), etiquetas.tolist())

    #X = np.mean(newcome, axis=1)

    prediccion = knn.predict([newcome])

    print()
    print("Usted dijo: ")
    if prediccion == 0:
        print('dale')
    elif prediccion == 1:
        print("derecha")
    elif prediccion == 2:
        print("izquierda")
    elif prediccion == 3:
        print("para")
    else:
        print("no reconocida")

    print()

#=====================================
# KNN BASADO EN LAS DISTANCIAS DADAS POR EL DTW, UNA DISTANCIA NULA INDICARIA QUE ES LA MISMA SEÑAL

def knnDTW(dtw):

    for i in range(dtw.shape[0]):
        print("DTW ", i, " - ", dtw[i, :])

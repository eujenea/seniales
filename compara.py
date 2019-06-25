from utils import saveMFCCToFile as save
from utils import readMFCCFromFile as read
import librosa as lib
import numpy as np
from sklearn import neighbors

def knnMFCC():
    file ="mfccFromAudio.npy"
    y = read(file)
    data = y[:,0]
    etiquetas = y[:,1] #.flatten().tolist()
    print(etiquetas)
    y, sr = lib.load('./audio.wav')
    mfccAudio = lib.feature.mfcc(y, sr,n_mfcc=30)

    audioMfccProm = np.mean(mfccAudio.T, axis=0)

    dataProm=np.zeros((data.shape[0],data[0].shape[0]))

    for idx in range(len(data)):
        # Probar mediana
        #dataProm.append(np.mean(commando,axis=0).flatten().tolist())
        commando=data[idx]
    
        dataProm[idx,:] = np.mean(commando.T,axis=0)


    # print(etiquetas)
    # print(type(etiquetas[0]))

    knn =  neighbors.KNeighborsClassifier(5)
    knn.fit(dataProm.tolist(), etiquetas.tolist())
    # print(np.mean(y[-1,0].T, axis=0))


    #X = np.mean(y[-1,0], axis=1)

    prediccion = knn.predict([audioMfccProm])

    print(prediccion)

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


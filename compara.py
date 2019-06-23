from utils import saveMFCCToFile as save
from utils import readMFCCFromFile as read
import numpy as np
from sklearn import neighbors

file ="mfccFromAudio.npy"
y = read(file)
data = y[:-1,0]
etiquetas = y[:-1,1] #.flatten().tolist()

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


X = np.mean(y[-1,0], axis=1)

prediccion = knn.predict([X])

print(prediccion)

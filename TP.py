import librosa as lib
import librosa.display as disp
import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np
from dtw import dtw

print("Iniciando")

# =====================================
# GRABACION DESDE MICROFONO
# Code here:


# =====================================
# COMPARATIVA

y1, sr1 = lib.load('./derecha_11k/derecha_1.wav')
# y2, sr2 = lib.load('./derecha_11k/derecha_5.wav')
# y2, sr2 = lib.load('./izquierda_11k/izquierda_4.wav')
# y2, sr2 = lib.load('./dale_11k/dale_1.wav')
y2, sr2 = lib.load('./para_11k/para_3.wav')
 
NMFCC=20 #NUMERO DE COEFICIENTES
plt.subplot(1, 3, 1)
mfcc1 = lib.feature.mfcc(y1, sr1,n_mfcc=NMFCC)
disp.specshow(mfcc1)

plt.subplot(1, 3, 2)
mfcc2 = lib.feature.mfcc(y2, sr2,n_mfcc=NMFCC)
disp.specshow(mfcc2)

dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=2))
print( 'Normalized distance between the two sounds:', dist)
print( 'muestra Mfcc 1:', mfcc1.shape )
print( 'muestra Mfcc 2:', mfcc2.shape )

plt.subplot(1,3,3)
plt.imshow(cost.T, origin='lower', interpolation='nearest',cmap='gray')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, cost.shape[0]-0.5))
plt.ylim((-0.5, cost.shape[1]-0.5))
plt.legend(str(dist))
plt.show()

# =====================================
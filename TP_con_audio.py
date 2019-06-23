import librosa as lib
import librosa.display as disp
import pyaudio
import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np
import wave
from dtw import dtw
from peakutils.plot import plot as pplot
from matplotlib import pyplot
print("Iniciando")

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
for i in range(int(20*44100/(1024*24))): #go for a few seconds
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    frame.append(data)
    #data
print("se grabó")
#lib.output.write_wav('./derecha_11k/audio.wav', cb, RATE)
  
stream.stop_stream()
stream.close()
p.terminate()

waveFile =wave.open('./derecha_11k/audio.wav','wb')
waveFile.setnchannels(1)
waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frame))
waveFile.close()

# =====================================
# COMPARATIVA

y1, sr1 = lib.load('./derecha_11k/audio.wav')
y2, sr2 = lib.load('./derecha_11k/derecha_1n.wav')
y3, sr3 = lib.load('./izquierda_11k/izquierda_4.wav')
y4, sr4 = lib.load('./dale_11k/dale_1.wav')
y5, sr5 = lib.load('./para_11k/para_3.wav')
 
NMFCC=15 #NUMERO DE COEFICIENTES

mfcc1 = lib.feature.mfcc(y1, sr1,n_mfcc=NMFCC)
mfcc2 = lib.feature.mfcc(y2, sr2,n_mfcc=NMFCC)
mfcc3 = lib.feature.mfcc(y3, sr3,n_mfcc=NMFCC)
mfcc4 = lib.feature.mfcc(y4, sr4,n_mfcc=NMFCC)
mfcc5 = lib.feature.mfcc(y5, sr5,n_mfcc=NMFCC)

dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=2))
dist1, cost1, acc_cost1, path1 = dtw(mfcc1.T, mfcc3.T, dist=lambda x, y: norm(x - y, ord=2))
dist2, cost2, acc_cost2, path3 = dtw(mfcc1.T, mfcc4.T, dist=lambda x, y: norm(x - y, ord=2))
dist3, cost3, acc_cost2, path3 = dtw(mfcc1.T, mfcc5.T, dist=lambda x, y: norm(x - y, ord=2))

print( 'Distancias entre sonidos')
print('derecha: ', dist, ' izquierda: ', dist1, 'dale: ', dist2, 'para: ', dist3)
if(dist < 30):
    print('dijiste derecha')
if(dist1 < 30):
    print('dijiste izquierda')
if(dist2 < 30):
    print('dijiste dale')
if(dist3 < 30):
    print('dijiste para') 
    
    

# =====================================
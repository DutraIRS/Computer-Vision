#%%
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import numpy as np
import cv2

#%% 1

filename = 'assets/StarWars60.wav'
data60, fs = sf.read(filename, dtype='int32')
#sd.play(data60, fs)
#sd.wait()

data10 = data60[:data60.shape[0]//6]
#sd.play(data10, fs)
#sd.wait()

datahat60 = np.fft.fft(data60)

plt.plot(data60)
plt.title('60 seconds sound spectrogram')
plt.show()
plt.plot(abs(datahat60))
plt.title('60 seconds fft spectrogram')
plt.show()

datahat10 = np.fft.fft(data10)

plt.plot(data10)
plt.title('10 seconds sound spectrogram')
plt.show()
plt.plot(abs(datahat10))
plt.title('10 seconds fft spectrogram')
plt.show()
#%% 2
def silence_high_freq(sound, pct_to_maintain):
    sound_fft = np.fft.fft(sound)

    forward_half = sound_fft[1:sound_fft.shape[0]//2+1]
    pnts_to_maintain = int(forward_half.shape[0] * pct_to_maintain)

    sound_fft[pnts_to_maintain+2:-pnts_to_maintain] = 0
    new_sound = np.fft.ifft(sound_fft).astype(sound.dtype)

    return new_sound

data10silent = silence_high_freq(data10, 0.1)
sd.play(data10silent, fs)
sd.wait()

# #%% IMAGEM
# import cv2 as cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread('..\\Imagens\\frutas.jpg')
# cv2.imshow('Frutas', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # %%
# impb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imghat = np.fft.fft2(impb)
# aux = np.abs(imghat)
# aux = aux/aux.max()
# cv2.imshow('fourier', aux*255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
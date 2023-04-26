#%%
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import numpy as np
import cv2

#%% 1
data60, fs = sf.read('assets/StarWars60.wav')
#sd.play(data60, fs)
#sd.wait()

data10 = data60[:data60.shape[0]//6]
sd.play(data10, fs)
sd.wait()


datahat10 = np.fft.fft(data10)

plt.plot(data10)
plt.title('10 seconds sound spectrogram')
plt.show()
plt.plot(abs(datahat10))
plt.title('10 seconds fft spectrogram')
plt.show()
#%% 2
def silence_high_freqs(sound, pct_to_maintain):
    sound_fft = np.fft.fft(sound)

    forward_half = sound_fft[1:sound_fft.shape[0]//2+1]
    pnts_to_maintain = int(forward_half.shape[0] * pct_to_maintain)

    sound_fft[pnts_to_maintain+2:-pnts_to_maintain] = 0
    new_sound = np.fft.ifft(sound_fft).astype(sound.dtype)

    return new_sound

data10silent = silence_high_freqs(data10, 0.1)
sd.play(data10silent, fs)
sd.wait()
#%% 3 a) Echo
sound, fs = sf.read('assets/WomanSinging.wav')

# 0.5 seconds of delay
delay = fs//2
time_padding = np.zeros(delay-1)

# play the original sound, then play the echo at 50% volume after the delay
echo_matrix = np.concatenate(([1], time_padding, [0.5]))
echoed_sound = np.convolve(sound, echo_matrix, mode="same")

sd.play(sound, fs)
sd.wait()
sd.play(echoed_sound, fs)
sd.wait()
#%% b) Reverb
sound, fs = sf.read('assets/WomanSinging.wav')

num_repeats = 10
decay_rate = 0.5

delay = sound.shape[0]//num_repeats
time_padding = np.zeros(delay-1)

reverb_matrix = np.array([1])

for i in range(num_repeats):
    reverb_matrix = np.concatenate((reverb_matrix, time_padding, [decay_rate**(i+1)]))

reverbed_sound = np.convolve(sound, reverb_matrix, mode="same")

sd.play(sound, fs)
sd.wait()
sd.play(reverbed_sound, fs)
sd.wait()

#%% 4
img = cv2.imread('assets/fruits.jpg')
impb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Fruits', impb)
cv2.waitKey(0)
cv2.destroyAllWindows()

imghat = np.fft.fft2(impb)

aux = np.abs(imghat)
aux = aux/aux.max()
cv2.imshow('fourier', aux*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

def cancel_high_freqs(img, pct_to_maintain):
    img_fft = np.fft.fft2(img)

    forward_half = img_fft[1:img_fft.shape[0]//2+1,
                           1:img_fft.shape[1]//2+1]
    pnts_to_maintain_x = int(forward_half.shape[0] * pct_to_maintain)
    pnts_to_maintain_y = int(forward_half.shape[1] * pct_to_maintain)

    img_fft[pnts_to_maintain_x+2:-pnts_to_maintain_x,
            pnts_to_maintain_y+2:-pnts_to_maintain_y] = 0
    new_img = np.fft.ifft2(img_fft).astype(img.dtype)

    return new_img/new_img.max()

def cancel_low_freqs(img, pct_to_maintain):
    img_fft = np.fft.fft2(img)

    forward_half = img_fft[1:img_fft.shape[0]//2+1,
                           1:img_fft.shape[1]//2+1]
    pnts_to_cancel_x = int(forward_half.shape[0] * (1 - pct_to_maintain))
    pnts_to_cancel_y = int(forward_half.shape[1] * (1 - pct_to_maintain))

    if (pnts_to_cancel_x == 0) and (pnts_to_cancel_y == 0):
        new_img = img
    else:
        img_fft[1:pnts_to_cancel_x+2,
                1:pnts_to_cancel_y+2] = 0
        img_fft[-pnts_to_cancel_x:,
                1:pnts_to_cancel_y+2] = 0
        img_fft[1:pnts_to_cancel_x+2,
                -pnts_to_cancel_y:] = 0
        img_fft[-pnts_to_cancel_x:,
                -pnts_to_cancel_y:] = 0    

        new_img = np.fft.ifft2(img_fft).astype(img.dtype)

    return new_img/new_img.max()

impb2 = cancel_high_freqs(impb, 0.1)
cv2.imshow('Fruits without high frequencies', impb2)
cv2.waitKey(0)
cv2.destroyAllWindows()

impb3 = cancel_low_freqs(impb2, 0.9)
cv2.imshow('Fruits without low frequencies', impb3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# i had the hypothesis that the black spots were due to zones of high brightness,
# where the high frequencies were more present, so i tried the following:
img = cv2.imread('assets/bald.jpg')
impb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Bald', impb)
cv2.waitKey(0)
cv2.destroyAllWindows()

impb2 = cancel_high_freqs(impb, 0.0075)
cv2.imshow('Bald without high frequencies', impb2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# which proves the hypothesis
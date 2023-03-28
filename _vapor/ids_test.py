import numpy as np
import scipy.io.wavfile as wavfile
import os


data_path = "/Volumes/T7/Multimodal/kinetics400/"
actions = ['playing_accordion', 'playing_bagpipes', 'playing_bass_guitar', 'playing_cello', 'playing_clarinet', 'playing_cymbals', 'playing_didgeridoo', 'playing_drums', 'playing_flute', 'playing_guitar', 'playing_harmonica', 'playing_harp', 'playing_keyboard', 'playing_organ', 'playing_piano', 'playing_recorder', 'playing_saxophone', 'playing_trombone', 'playing_trumpet', 'playing_ukulele', 'playing_violin', 'playing_xylophone']
files = ['IMF1juHAXYs', 'fJdbLkikmqw']


ids = np.load("data_npy/kinetics400/kinetics400_ids_instruments.npy")
print(ids[[392, 886]])

for action in actions:
    list = os.listdir("%s%s" % (data_path, action))
    for file in files:
        if "%s_audio.mp4" % file in list:
            print(action)
            print("YAY")
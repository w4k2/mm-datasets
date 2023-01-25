import numpy as np
import matplotlib.pyplot as plt
from time import sleep

datasets = [
    ['playing_badminton', 'playing_basketball', 'playing_cricket', 'playing_ice_hockey', 'playing_kickball', 'playing_paintball', 'playing_squash_or_racquetball', 'playing_tennis', 'playing_volleyball'],
    
    ['playing_accordion', 'playing_bagpipes', 'playing_bass_guitar', 'playing_cello', 'playing_clarinet', 'playing_cymbals', 'playing_didgeridoo', 'playing_drums', 'playing_flute', 'playing_guitar', 'playing_harmonica', 'playing_harp', 'playing_keyboard', 'playing_organ', 'playing_piano', 'playing_recorder', 'playing_saxophone', 'playing_trombone', 'playing_trumpet', 'playing_ukulele', 'playing_violin', 'playing_xylophone'],
    
    
    ['riding_a_bike', 'riding_camel', 'riding_elephant', 'riding_mechanical_bull', 'riding_mountain_bike', 'riding_mule', 'riding_or_walking_with_horse', 'riding_scooter', 'riding_unicycle'],
    
    ['belly_dancing', 'breakdancing', 'country_line_dancing', 'dancing_ballet', 'dancing_charleston', 'dancing_gangnam_style', 'dancing_macarena', 'jumpstyle_dancing', 'robot_dancing', 'salsa_dancing', 'swing_dancing', 'tango_dancing', 'tap_dancing'],
    
    ['eating_burger', 'eating_cake', 'eating_carrots', 'eating_chips', 'eating_doughnuts', 'eating_hotdog', 'eating_ice_cream', 'eating_spaghetti', 'eating_watermelon']
]

dataset_names = ["sport", "instruments", "riding", "dancing", "eating"]

for dataset_id, dataset in enumerate(dataset_names):
    X_video = np.load("data_npy/kinetics400/kinetics400_video_%s.npy" % dataset)
    X_audio = np.load("data_npy/kinetics400/kinetics400_audio_%s.npy" % dataset)
    y = np.load("data_npy/kinetics400/kinetics400_y_%s.npy" % dataset)
    ids = np.load("data_npy/kinetics400/kinetics400_ids_%s.npy" % dataset)
    
    print(dataset)
    print(X_video.shape)
    print(X_audio.shape)
    print(y.shape)
    print(np.unique(y, return_counts=True))
    print(ids.shape)
    # exit()
    
    for task in range(1200, 1250):
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        ax[0].imshow(X_video[task])
        ax[1].imshow(X_audio[task])
        ax[0].set_title(datasets[0][y[task]], fontsize=20)
    
        plt.tight_layout()
        plt.savefig("foo.png")
        sleep(.2)
        plt.close()
        exit()
    
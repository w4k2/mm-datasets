"""
NOPE Whole mmIMDb dataset to npy + Resnet18 transforms.
"""

import os
from pathlib import Path
import numpy as np


data_path = "/Volumes/T7/Multimodal/kinetics400/"
files = os.listdir(data_path)
files = list(set([file.split(".")[0] for file in files]))
files.remove("")
files.sort()

# View all classes with a given action
selected_actions = ["playing", "riding", "climbing", "cooking", "dancing", "eating", "making"]
for action in selected_actions:
    action_idxs = np.argwhere(np.array([action in a for a in files]) == True).flatten()
    action_names = np.array(files)[action_idxs]
    # print(action_names)

datasets = [
    ['playing_badminton', 'playing_basketball', 'playing_cricket', 'playing_ice_hockey', 'playing_kickball', 'playing_paintball', 'playing_squash_or_racquetball', 'playing_tennis', 'playing_volleyball'],
    
    ['playing_accordion', 'playing_bagpipes', 'playing_bass_guitar', 'playing_cello', 'playing_clarinet', 'playing_cymbals', 'playing_didgeridoo', 'playing_drums', 'playing_flute', 'playing_guitar', 'playing_harmonica', 'playing_harp', 'playing_keyboard', 'playing_organ', 'playing_piano', 'playing_recorder', 'playing_saxophone', 'playing_trombone', 'playing_trumpet', 'playing_ukulele', 'playing_violin', 'playing_xylophone'],
    
    ['clay_pottery_making', 'making_a_cake', 'making_a_sandwich', 'making_bed', 'making_jewelry', 'making_pizza', 'making_snowman', 'making_sushi', 'making_tea'],
    
    ['riding_a_bike', 'riding_camel', 'riding_elephant', 'riding_mechanical_bull', 'riding_mountain_bike', 'riding_mule', 'riding_or_walking_with_horse', 'riding_scooter', 'riding_unicycle'],
    
    ['belly_dancing', 'breakdancing', 'country_line_dancing', 'dancing_ballet', 'dancing_charleston', 'dancing_gangnam_style', 'dancing_macarena', 'jumpstyle_dancing', 'robot_dancing', 'salsa_dancing', 'swing_dancing', 'tango_dancing', 'tap_dancing'],
    
    ['eating_burger', 'eating_cake', 'eating_carrots', 'eating_chips', 'eating_doughnuts', 'eating_hotdog', 'eating_ice_cream', 'eating_spaghetti', 'eating_watermelon']
]
dataset_names = ["playing_sport", "playing_instruments", "making", "riding", "dancing", "eating"]
# print("\n")
# How many instances
# for id, dataset in enumerate(datasets):
#     total = 0
#     for action in dataset:
#         files = os.listdir(data_path + "%s/" % action)
#         print(action)
#         total += len(files)
#     print("%s dataset has %i instances!\n" % (dataset_names[id], int(total/2)))
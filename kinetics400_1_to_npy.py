"""
NOPE Whole mmIMDb dataset to npy + Resnet18 transforms.
"""

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from moviepy.editor import *
from torchvision.models import ResNet18_Weights
from torch import from_numpy
import scipy.io.wavfile as wavfile
import librosa, librosa.display
from cv2 import resize


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
    
    
    ['riding_a_bike', 'riding_camel', 'riding_elephant', 'riding_mechanical_bull', 'riding_mountain_bike', 'riding_mule', 'riding_or_walking_with_horse', 'riding_scooter', 'riding_unicycle'],
    
    ['belly_dancing', 'breakdancing', 'country_line_dancing', 'dancing_ballet', 'dancing_charleston', 'dancing_gangnam_style', 'dancing_macarena', 'jumpstyle_dancing', 'robot_dancing', 'salsa_dancing', 'swing_dancing', 'tango_dancing', 'tap_dancing'],
    
    ['eating_burger', 'eating_cake', 'eating_carrots', 'eating_chips', 'eating_doughnuts', 'eating_hotdog', 'eating_ice_cream', 'eating_spaghetti', 'eating_watermelon']
    
    # ['clay_pottery_making', 'making_a_cake', 'making_a_sandwich', 'making_bed', 'making_jewelry', 'making_pizza', 'making_snowman', 'making_sushi', 'making_tea'],
]
# dataset_names = ["sport", "instruments", "making", "riding", "dancing", "eating"]
dataset_names = ["sport", "instruments", "riding", "dancing", "eating"]
# print("\n")
# How many instances
# for id, dataset in enumerate(datasets):
#     total = 0
#     for action in dataset:
#         files = os.listdir(data_path + "%s/" % action)
#         # print(action)
#         total += len(files)
#     print("%s dataset has %i instances!\n" % (dataset_names[id], int(total/2)))

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

for dataset_id, dataset in enumerate(datasets):
    print(dataset_names[dataset_id])
    X_video = []
    X_audio = []
    y = []
    
    for action_id, action in enumerate(dataset):
        print(action)
        files = os.listdir(data_path + "%s/" % action)
        files = list(set([file[:-10] for file in files]))
        files.sort()
        
        for file in tqdm(files):
            video_path = "%s%s/%s_video.mp4" % (data_path, action, file)
            audio_path = "%s%s/%s_audio.mp4" % (data_path, action, file)
            if Path(video_path).exists() and Path(audio_path).exists():
                # print("Sa obie!")
                # Video
                video = VideoFileClip(video_path, audio=False)
                all_frames = []
                for id, frame in enumerate(video.iter_frames()):
                    resnet18_weights = ResNet18_Weights.IMAGENET1K_V1
                    resnet18_transforms = resnet18_weights.transforms()
                    frame = np.swapaxes(resnet18_transforms(from_numpy(np.swapaxes(np.array(frame), 0, 2))).numpy(), 0, 2)
                    fimg = np.mean(frame, axis=2)
                    fimg = np.fft.fft2(fimg)
                    fimg = np.fft.fftshift(fimg)
                    all_frames.append(fimg)
                    
                all_frames = np.moveaxis(np.array(all_frames), 0, 2)
                extracted = np.mean(all_frames, axis=2)
                mask = create_circular_mask(extracted.shape[0], extracted.shape[1], radius=int(extracted.shape[0]/4))
                extracted[mask==False] = 1
                resnet_fourier = np.stack((np.abs(extracted), np.abs(extracted), np.abs(extracted)), axis=2).astype(np.uint8)
                X_video.append(resnet_fourier)
                
                # Audio
                if not Path("%s%s/%s_audio.wav" % (data_path, action, file)).exists():
                    os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(audio_path, audio_path[:-4]))
                
                sr, signal = wavfile.read("%s%s/%s_audio.wav" % (data_path, action, file))
                signal = np.mean(signal, axis=1)
                
                # Mel-Spectrogram
                n_fft = 1024
                hop_length = 256
                audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
                mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
                spectrogram = np.abs(mel_signal)
                power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
                mel_spec_img = np.flip(power_to_db, axis=0)
                mel_spec_img_standard = (mel_spec_img - np.mean(mel_spec_img)) / np.std(mel_spec_img)
                mel_spec_img_standard = resize(mel_spec_img_standard, (624, 128))
                mel_spec_img = np.stack((mel_spec_img_standard, mel_spec_img_standard, mel_spec_img_standard), axis=2)
                X_audio.append(mel_spec_img)
                
                y.append(action_id)
                
    X_video = np.array(X_video)
    X_audio = np.array(X_audio)
    y = np.array(y)
    np.save("data_npy/kinetics400/kinetics400_video_%s" % dataset_names[dataset_id], X_video)
    np.save("data_npy/kinetics400/kinetics400_audio_%s" % dataset_names[dataset_id], X_audio)
    np.save("data_npy/kinetics400/kinetics400_y_%s" % dataset_names[dataset_id], y)
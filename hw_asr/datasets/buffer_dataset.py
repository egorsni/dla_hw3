import logging
from pathlib import Path
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.utils.parse_config import ConfigParser
from text import text_to_sequence
import time
from tqdm import tqdm
import os
import numpy as np
import librosa
import pyworld as pw
from audio import hparams_audio
import numpy as np

logger = logging.getLogger(__name__)

def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


class BufferDataset(Dataset):
    def __init__(
            self,
            data_path, alignment_path, wav_path, mel_ground_truth, text_cleaners,
            *args, **kwargs
    ):
        self.data_path = data_path
        self.alignment_path = alignment_path
        self.mel_ground_truth = mel_ground_truth
        self.text_cleaners = text_cleaners
        self.wav_path = wav_path
        
        self.buffer = self.get_data_to_buffer()
        self.length_dataset = len(self.buffer)
        
    def get_data_to_buffer(self):
        buffer = list()
        text = process_text(self.data_path)
    
        start = time.perf_counter()
        wav_names = os.listdir(self.wav_path)
        wav_names = sorted(wav_names)
        Path("./data/pitch").mkdir(parents=True, exist_ok=True)
        Path("./data/energy").mkdir(parents=True, exist_ok=True)
        pitchs = []
        energys = []
        for i in tqdm(range(len(text))):
    
            mel_gt_name = os.path.join(
                self.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
            mel_gt_target = np.load(mel_gt_name)
            duration = np.load(os.path.join(
                self.alignment_path, str(i)+".npy"))
            character = text[i][0:len(text[i])-1]
            character = np.array(
                text_to_sequence(character, [self.text_cleaners]))
    
            character = torch.from_numpy(character)
            duration = torch.from_numpy(duration)
            mel_gt_target = torch.from_numpy(mel_gt_target)
            
            if not os.path.isfile(f"./data/pitch/{i}.npy") or not os.path.isfile(f"./data/energy/{i}.npy"):
                wav, sr = librosa.load(os.path.join(self.wav_path, wav_names[i]))
                energy = torch.stft(torch.tensor(wav), n_fft=1024, win_length=hparams_audio.win_length, hop_length=hparams_audio.hop_length).transpose(0,1)
                energy = torch.linalg.norm(energy, dim=-1)
                energy = torch.linalg.norm(energy, dim=-1)
                pitch, t = pw.dio(wav.astype(np.float64), sr, frame_period=hparams_audio.hop_length * 1000 / sr)
                pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr).astype(np.float32)
                np.save(f"./data/pitch/{i}.npy", pitch)
                np.save(f"./data/energy/{i}.npy", energy)
            else:
                pitch = np.load(f"./data/pitch/{i}.npy")
                energy = np.load(f"./data/energy/{i}.npy")
    
            pitchs.append(pitch)
            energys.append(energy)
            buffer.append({"text": character, "duration": duration,
                           "mel_target": mel_gt_target,
                            "pitch":torch.from_numpy(pitch),
                              "energy": torch.from_numpy(energy)})
    
        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end-start))
#         print('max_pitch', np.max(pitchs), 'min pitch', np.min(pitchs))
#         print('max_energy', np.max(energys), 'min energy', np.min(energys))
        return buffer

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __len__(self):
        return self.length_dataset
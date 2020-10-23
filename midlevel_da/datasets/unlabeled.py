import torch
from torch.utils.data import Dataset
from utils import *
from datasets.dataset_utils import slice_func, normalize_spec, get_dataset_stats
from helpers.audio import MadmomAudioProcessor
from helpers.specaugment import SpecAugment
from paths import *
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger()


class UnlabeledAudioDataset(Dataset):
    def __init__(self, cache_dir=path_cache_fs,
                 name='audio_dataset',
                 audio_files=None,
                 duration=15,
                 aud_processor=None,
                 audio_dir=None,
                 return_mode='spec',
                 slice_mode='random', **kwargs):
        self.dset_name = name
        if aud_processor is not None:
            self.processor = aud_processor
        else:
            self.processor = MadmomAudioProcessor(fps=31.3)

        self.dataset_cache_dir = os.path.join(cache_dir, name)
        self.duration = duration  # seconds
        self.audio_files = audio_files
        self.audio_dir = audio_dir
        self.mode = return_mode
        self.slice_mode = slice_mode

        if isinstance(kwargs.get('augment'), SpecAugment):
            self.augment = kwargs['augment']
        elif kwargs.get('augment') == 'none' or kwargs.get('augment') is None:
            self.augment = lambda x: x
        else:
            logger.info(f"WARNING: No spec augment function assigned -- got {kwargs.get('augment')}; should be SpecAugment instance or None or 'none'!")
            self.augment = lambda x: x

        if kwargs.get('normalizing_dset'):
            logger.info(f"WARNING (UnlabeledAudioDataset - {self.dset_name}): Using {kwargs.get('normalizing_dset')} mean and std using default audio_processor parameters!")
            self.norm_mean, self.norm_std = get_dataset_stats(kwargs.get('normalizing_dset'))

        self.kwargs = kwargs

    def __getitem__(self, ind):
        audio_path = self.audio_files[ind]
        song_name = os.path.basename(audio_path)

        x = self._get_spectrogram(audio_path, self.dataset_cache_dir).spec
        if self.duration is not None:
            slice_length = self.processor.times_to_frames(self.duration)
            x_sliced, start_time, end_time = slice_func(x, slice_length, self.processor, mode=self.slice_mode)
        else:
            x_sliced = x
            start_time = 0
            end_time = self.processor.frames_to_times(len(x))

        if self.kwargs.get('normalizing_dset'):
            x_sliced = normalize_spec(x_sliced, mean=self.norm_mean, std=self.norm_std)
        else:
            x_sliced = normalize_spec(x_sliced, dset_name=self.dset_name, aud_processor=self.processor)

        # x_sliced = (x_sliced - np.mean(x_sliced))/np.std(x_sliced)

        x_aug = self.augment(x_sliced).astype(np.float32)
        x_aug = torch.from_numpy(x_aug)
        return audio_path, x_aug

    def _get_spectrogram(self, audio_path, dataset_cache_dir):
        if self.kwargs.get('aud_path_type') == 'absolute':
            specpath = os.path.join(dataset_cache_dir, self.processor.get_params.get("name"), str(os.path.basename(audio_path).split('.')[0]) + '.specobj')
        elif self.kwargs.get('aud_path_type') == 'relative':
            specpath = os.path.join(dataset_cache_dir, self.processor.get_params.get("name"), str(audio_path.split('.')[0]) + '.specobj')
            audio_path = os.path.join(self.kwargs.get('aud_path_prefix'), audio_path)
        else:
            specpath = os.path.join(dataset_cache_dir, self.processor.get_params.get("name"),
                                    str(os.path.basename(audio_path).split('.')[0]) + '.specobj')

        specdir = os.path.split(specpath)[0]
        if not os.path.exists(specdir):
            print(specdir, os.path.exists(specdir))
            os.makedirs(specdir)
        try:
            return pickleload(specpath)
        except:
            print(f"Calculating spectrogram for {audio_path} and saving to {specpath}")
            if self.audio_dir is not None:
                audio_path = os.path.join(self.audio_dir, audio_path)
            spec_obj = self.processor(audio_path)
            pickledump(spec_obj, specpath)
            return spec_obj

    def __len__(self):
        return len(self.audio_files)

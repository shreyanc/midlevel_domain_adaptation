import hashlib
import os

from tqdm import tqdm

from helpers.audio import MadmomAudioProcessor
from paths import *
from utils import *
import torch
import torch.utils.data
import numpy as np
from sklearn.model_selection import train_test_split

dset_stats = {}


def get_dataset_stats(dset_name, aud_processor=None):
    global dset_stats
    if aud_processor is None:
        aud_processor = MadmomAudioProcessor(fps=31.3)
    dataset_cache_dir = os.path.join(path_cache_fs, dset_name)
    all_specs_path = os.path.join(dataset_cache_dir, aud_processor.get_params.get("name"))

    if dset_stats.get(dset_name) is not None:
        try:
            mean = dset_stats[dset_name]['mean']
            std = dset_stats[dset_name]['std']
        except KeyError:
            raise Exception(f"mean or std not found for {dset_name}")
    else:
        try:
            mean = np.load(os.path.join(all_specs_path, 'mean.npy'))
        except FileNotFoundError:
            all_specs_list = list_files_deep(all_specs_path, full_paths=True, filter_ext=['.specobj'])
            mean = 0.0
            for specobj in tqdm(all_specs_list, desc=f'Calculating mean for {dset_name} {aud_processor.get_params.get("name")}'):
                spec = pickleload(specobj).spec
                mean += np.mean(spec).item()
            mean = mean / len(all_specs_list)
            np.save(os.path.join(all_specs_path, 'mean.npy'), mean)

        try:
            std = np.load(os.path.join(all_specs_path, 'std.npy'))
        except FileNotFoundError:
            all_specs_list = list_files_deep(all_specs_path, full_paths=True, filter_ext=['.specobj'])
            sum_of_mean_of_squared_dev = 0.0
            for specobj in tqdm(all_specs_list, desc=f'Calculating std for {dset_name} {aud_processor.get_params.get("name")}'):
                spec = pickleload(specobj).spec
                sum_of_mean_of_squared_dev += np.mean(np.square(spec - mean)).item()
            std = np.sqrt(sum_of_mean_of_squared_dev / len(all_specs_list))
            np.save(os.path.join(all_specs_path, 'std.npy'), std)

        dset_stats[dset_name] = {'mean': mean, 'std': std}

    return mean, std


def normalize_spec(spec, mean=None, std=None, dset_name=None, aud_processor=None):
    if mean is None and std is None:
        mean, std = get_dataset_stats(dset_name, aud_processor)
    assert (isinstance(mean, np.ndarray) and isinstance(std, np.ndarray)) or (isinstance(mean, float) and isinstance(std, float)), \
        print(f"Either mean or std is not a float: mean={mean}, std={std}")
    return (spec - mean) / std


def slice_func(spec, length, processor=None, mode='random', offset_seconds=0, slice_times=None):
    if slice_times is not None:
        start_time, end_time = slice_times[0], slice_times[1]
        return spec[:, processor.times_to_frames(start_time): processor.times_to_frames(end_time)], start_time, end_time

    offset_frames = int(processor.times_to_frames(offset_seconds))

    length = int(length)

    while spec.shape[-1] < offset_frames + length:
        spec = np.append(spec, spec[:, :length - spec.shape[-1]], axis=1)
    xlen = spec.shape[-1]

    midpoint = xlen // 2 + offset_frames

    if mode == 'start':
        start_time = processor.frames_to_times(offset_frames)
        end_time = processor.frames_to_times(offset_frames + length)
        output = spec[:, offset_frames: offset_frames + length]
    elif mode == 'end':
        start_time = processor.frames_to_times(xlen - length)
        end_time = processor.frames_to_times(xlen)
        output = spec[:, -length:]
    elif mode == 'middle':
        start_time = processor.frames_to_times(xlen - length)
        end_time = processor.frames_to_times(xlen)
        output = spec[:, midpoint - length // 2: midpoint + length // 2 + 1]
    elif mode == 'random':
        k = torch.randint(offset_frames, xlen - length + 1, (1,))[0].item()
        start_time = processor.frames_to_times(k)
        end_time = processor.frames_to_times(k + length)
        output = spec[:, k: k + length]
    else:
        raise Exception(f"mode must be in ['start', 'end', 'middle', 'random'], is {mode}")

    return output, start_time, end_time


class DsetNoLabel(torch.utils.data.Dataset):
    # Make sure that your dataset actually returns many elements!
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        ret_stuff = self.dset[index]
        return ret_stuff[:-1] if len(ret_stuff) > 2 else ret_stuff

    def __len__(self):
        return len(self.dset)


class DsetMultiDataSources(torch.utils.data.Dataset):
    def __init__(self, *dsets):
        self.dsets = dsets
        self.lengths = [len(d) for d in self.dsets]

    def __getitem__(self, index):
        return_triplets = []
        for ds in self.dsets:
            idx = index % len(ds)
            try:
                path, x, y = ds[idx]
                return_triplets.append((path, x, y))
            except:
                # if dataset does not return path, generate a (semi-)unique hash from the sum of the tensor, to be used as an identifier of the tensor for caching
                x, y = ds[idx]
                return_triplets.append((hashlib.md5(f"{str(torch.sum(x).item())}".encode("UTF-8")).hexdigest(), x, y))

        return tuple(return_triplets)

    def __len__(self):
        return min(self.lengths)


class DsetThreeChannels(torch.utils.data.Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        image, label = self.dset[index]
        return image.repeat(3, 1, 1), label

    def __len__(self):
        return len(self.dset)


if __name__ == '__main__':
    print(get_dataset_stats('midlevel'))

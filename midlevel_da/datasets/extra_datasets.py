from sklearn.model_selection import train_test_split

from paths import *
from utils import *
from datasets.unlabeled import UnlabeledAudioDataset


def get_extra_dataset(dset_name, seed=None, augment=None, test_size=0.1, labeled=False, normalizing_dset=None):
    if seed is None:
        seed = 0

    if dset_name == 'maestro':
        domain_files = list_files_deep(path_maestro_audio_15sec, filter_ext=['.wav', '.WAV', '.mp3'], full_paths=True)
        tg_train_files, tg_test_files = train_test_split(domain_files, test_size=test_size, random_state=seed)
        tg_tr_dataset = UnlabeledAudioDataset(name='maestro', audio_files=tg_train_files, augment=augment, slice_mode='start', normalizing_dset=normalizing_dset)
        tg_te_dataset = UnlabeledAudioDataset(name='maestro', audio_files=tg_test_files, augment=None, slice_mode='start', normalizing_dset=normalizing_dset)
        return tg_tr_dataset, tg_te_dataset

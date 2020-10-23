import getpass
import os
import re
rk = re.compile("rechenknecht[0-8].cp.jku.at")

hostname = os.uname()[1]
username = getpass.getuser()

use_local = False

if bool(rk.match(hostname)):
    local_datasets_root = '/home/shreyan/shared/datasets/'
    local_home_root = '/home/shreyan/'
    local_run_dir = '/home/shreyan/RUNS/'
    
    fs_datasets_root = '/share/cp/datasets/'
    fs_home_root = '/share/home/shreyan/'
    fs_run_dir = '/share/home/shreyan/RUNS/'

elif hostname == 'shreyan-All-Series':
    local_datasets_root = '/mnt/2tb/datasets/'
    local_home_root = ''
    local_run_dir = ''

    fs_datasets_root = '/home/shreyan/mounts/fs/datasets@fs/'
    fs_home_root = '/home/shreyan/mounts/fs/home@fs/'
    fs_run_dir = '/home/shreyan/mounts/fs/home@fs/RUNS/'

elif hostname == 'shreyan-HP':
    local_datasets_root = '/home/shreyan/mounts/pc/datasets/'
    local_home_root = ''
    local_run_dir = ''

    fs_datasets_root = '/home/shreyan/mounts/fs/datasets@fs/'
    fs_home_root = ''
    fs_run_dir = '/home/shreyan/mounts/fs/home@fs/RUNS/'
else:
    local_datasets_root = '/home/shreyan/mounts/pc/datasets/'
    local_home_root = ''
    local_run_dir = ''

    fs_datasets_root = '/home/shreyan/mounts/fs/datasets@fs/'
    fs_home_root = ''
    fs_run_dir = '/home/shreyan/mounts/fs/home@fs/RUNS/'

if use_local:
    MAIN_RUN_DIR = local_run_dir
    DATASETS_ROOT = local_datasets_root
    HOME_ROOT = local_home_root
else:
    MAIN_RUN_DIR = fs_run_dir
    DATASETS_ROOT = fs_datasets_root
    HOME_ROOT = fs_home_root

path_cache_fs = os.path.join(DATASETS_ROOT, 'shreyan_data_caches')

path_midlevel_annotations_dir = os.path.join(DATASETS_ROOT, 'MidlevelFeatures/metadata_annotations')
path_midlevel_annotations = os.path.join(DATASETS_ROOT, 'MidlevelFeatures/metadata_annotations/annotations.csv')
path_midlevel_metadata = os.path.join(DATASETS_ROOT, 'MidlevelFeatures/metadata_annotations/metadata.csv')
path_midlevel_metadata_piano = os.path.join(DATASETS_ROOT, 'MidlevelFeatures/metadata_annotations/metadata_piano.csv')
path_midlevel_audio_dir = os.path.join(DATASETS_ROOT, 'MidlevelFeatures/audio')

path_maestro_audio = os.path.join(DATASETS_ROOT, 'maestro-v2.0.0/audio_and_midi')
path_maestro_audio_15sec = os.path.join(DATASETS_ROOT, 'maestro-v2.0.0/audio_15sec')

path_ce_audio_15sec = os.path.join(DATASETS_ROOT, 'con_espressione_game/data_audio_15sec')
path_ce_audio = os.path.join(DATASETS_ROOT, 'con_espressione_game/audio')
path_ce_metadata = os.path.join(DATASETS_ROOT, 'con_espressione_game/metadata.csv')
path_ce_root = os.path.join(DATASETS_ROOT, 'con_espressione_game')

path_ce_public_root = os.path.join(DATASETS_ROOT, 'con_espressione_game_dataset_(public)')

import sys

from torch.utils.data import DataLoader

from datasets.extra_datasets import get_extra_dataset
from datasets.midlevel import *
from helpers.train_funcs import test, train_da_backprop, train
from models.model_configs import config_cp_field_shallow_m2

PROJECT_NAME = 'midlevel_da'
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.append(PROJECT_ROOT)
SUBPROJECT_NAME = 'main_no_da'

import hashlib
import time
import logging
from torch import nn, optim
from sklearn.model_selection import train_test_split
from models.cpresnet import CPResnet_BackProp, CPResnet

from datetime import datetime as dt
from utils import *
from paths import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
writer = None
SEED = int(int(time.time() * 10e3) - int(time.time()) * 10e3)
rState = np.random.RandomState(seed=SEED)
train_valid_split = train_test_split
ml_names = ['melody', 'articulation', 'rhythm_complexity', 'rhythm_stability', 'dissonance', 'tonal_stability', 'minorness']
NUM_WORKERS = 8 if torch.cuda.is_available() else 0

dtstr = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
hash = hashlib.sha1()
hash.update(str(time.time()).encode('utf-8'))
run_hash = hash.hexdigest()[:5]
run_name = f'{run_hash}_{dtstr}'

if os.uname()[1] in ['shreyan-HP', 'shreyan-All-Series']:
    PROJECT_RUN_DIR = os.path.join(MAIN_RUN_DIR, '_debug_runs')
else:
    PROJECT_RUN_DIR = os.path.join(MAIN_RUN_DIR, SUBPROJECT_NAME)
if not os.path.exists(os.path.join(PROJECT_RUN_DIR, run_name)):
    os.makedirs(os.path.join(PROJECT_RUN_DIR, run_name))

curr_run_dir = os.path.join(PROJECT_RUN_DIR, run_name)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(curr_run_dir, f'{run_name}.log'))
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)


def main():
    # HPARAMS
    h = dict(domain_split='piano',
             batch_size=8,
             learning_rate=1e-3,
             patience=14,
             input_noise='none',
             label_noise=0.0,
             )
    log_dict(logger, h, delimiter=', ')
    logger.info(f"SEED = {SEED}")
    # -------------------------------------------------

    # INIT MODEL
    net = CPResnet(config=config_cp_field_shallow_m2, num_targets=7).to(device)

    # INIT DATA
    tr_ids, te_ids = load_midlevel_aljanaki(tsize=0.1)
    # 20% of eval data as validation set, 80% as test set
    te_ids, va_ids = train_valid_split(te_ids, test_size=int(round(0.2 * len(te_ids))), random_state=SEED)

    tr_dataset = MidlevelDataset(select_song_ids=tr_ids, augment=None)
    va_dataset = MidlevelDataset(select_song_ids=va_ids)
    te_dataset = MidlevelDataset(select_song_ids=te_ids)

    logger.info(f"LENGTHS: sc_tr={len(tr_dataset)}, sc_va={len(va_dataset)}, tg_te={len(te_dataset)}")

    tr_dataloader = DataLoader(tr_dataset, batch_size=h['batch_size'], shuffle=True, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)
    va_dataloader = DataLoader(va_dataset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)
    te_dataloader = DataLoader(te_dataset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)

    # INIT TRAINER
    optimizer = optim.Adam(net.parameters(), lr=h['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 10, 15, 20], gamma=0.5)
    criterion = nn.MSELoss().to(device)
    es = EarlyStopping(patience=h['patience'], condition='maximize', verbose=True,
                       save_dir=os.path.join(curr_run_dir, 'saved_models'),
                       saved_model_name=net.name + '_' + run_name[:5] + '_teacher')

    test_metric = 'corr_avg'

    for epoch in range(1, 100):
        train(net, tr_dataloader, optimizer, criterion, epoch, writer, run_name + f'_epoch-{epoch}', dataloader_len=len(tr_dataloader))
        scheduler.step()
        val_corr = test(net, va_dataloader, criterion, writer, epoch, mets=[test_metric])[0][test_metric]
        test_corr = test(net, te_dataloader, criterion, writer, epoch, mets=[test_metric])[0]

        logger.info(
            f"Epoch {epoch} sc val {test_metric} = {round(val_corr, 4)}, sc test {test_metric} = {round(test_corr[test_metric], 4)}")

        es(val_corr, net)
        if es.early_stop:
            logger.info(f"Early stop - trained for {epoch - es.counter} epochs - best metric {es.best_score}")
            break

    load_model(es.save_path, net)
    tg_test_corr = test(net, te_dataloader, criterion, writer, epoch=-1, mets=[test_metric])[0][test_metric]
    logger.info(f"Final target {test_metric} = {round(tg_test_corr, 4)}")


if __name__ == '__main__':
    main()

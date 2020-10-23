import torch
import torch.nn as nn

from models.model_configs import *


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        module.bias.data.zero_()


model_config = config_cp_field_shallow_m2
dilat_limit = 0
dilat_start = 0


def _getk(i):
    k = i
    nblock_per_stage = (model_config['depth'] - 2) // 6
    i = (k - 1) // (nblock_per_stage * 2)
    "stage%d" % (i + 1), nblock_per_stage, 'k%ds' % ((k + 1) % 2 + 1), ((k - 1) % (nblock_per_stage * 2)) // 2
    ke = model_config["stage%d" % (i + 1)]['k%ds' % ((k + 1) % 2 + 1)][((k - 1) % (nblock_per_stage * 2)) // 2]
    if k < dilat_start or k > dilat_limit:
        return ke
    return (ke - 1) * 2 + 1


def _gets(i):
    k = i
    if k % 2 == 1:
        return 1
    nblock_per_stage = (model_config['depth'] - 2) // 6
    i = (k - 1) // (nblock_per_stage * 2)
    "stage%d" % (i + 1), nblock_per_stage, 'k%ds' % ((k + 1) % 2 + 1), (k % (nblock_per_stage * 2)) // 2
    if (((k - 1) % (nblock_per_stage * 2)) // 2 + 1) in set(model_config["stage%d" % (i + 1)]['maxpool']):
        return 2
    return 1


def get_maxrf(i):
    if i == 0:
        return 2, 5  # starting RF
    s, rf = get_maxrf(i - 1)
    s = s * _gets(i)
    rf = rf + (_getk(i) - 1) * s
    return s, rf


if __name__ == '__main__':
    print(get_maxrf(6))
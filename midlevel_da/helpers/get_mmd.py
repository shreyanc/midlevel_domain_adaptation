import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_centroid(dataloader, model):
    all_outputs = []
    for batch_idx, (_, inputs, labels) in enumerate(dataloader):
        inputs = inputs.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            try:
                all_outputs.append(outputs.cpu())
            except:
                all_outputs.append(outputs['output'].cpu())
    return torch.mean(torch.cat(all_outputs), dim=0)


def get_mmd(loader_1, loader_2, model):
    print("calculating mmd")
    model.eval()
    centroid_1 = get_centroid(loader_1, model)
    centroid_2 = get_centroid(loader_2, model)
    model.train()
    return torch.dist(centroid_1, centroid_2, 2).item()


def mmd_select_naive(mmd):
    return np.argmin(mmd)


def mmd_select_scale(mmd, sce):
    sce = np.asarray(sce)
    mmd = np.asarray(mmd)
    scl = np.min(sce) / np.min(mmd)
    return np.argmin(sce + mmd * scl)

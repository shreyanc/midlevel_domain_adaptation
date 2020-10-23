import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import logging

from utils import compute_metrics

logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, optimizer, criterion, epoch, writer, run_name, **kwargs):
    model.train()
    loss_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=False, total=len(dataloader), desc=run_name):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)
        optimizer.zero_grad()
        output = model(inputs)
        try:
            loss = criterion(output['output'].float(), labels.float())
        except Exception as e:
            print(e)
            loss = criterion(output.float(), labels.float())
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    epoch_loss = np.mean(loss_list)

    writer.add_scalar(f"{kwargs.get('phase')}/training_loss", epoch_loss, epoch)

    return {'avg_loss': epoch_loss}


def train_da_backprop(model, dataloader, optimizer, criterion, epoch, writer, run_name, **kwargs):
    model.train()
    loss_list = []
    ml_loss_list = []
    domain_loss_list = []

    dlen = kwargs['dataloader_len']

    p = epoch / 20
    lambda_ = 2. / (1 + np.exp(-10 * p)) - 1
    # lambda_ *= 0.1

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=dlen, desc=run_name):
        (song_ids_ml, inputs_ml, labels), (song_names_p, inputs_piano) = batch
        inputs_ml, labels, inputs_piano = inputs_ml.to(device), labels.to(device), inputs_piano.to(device)
        inputs = torch.cat([inputs_ml, inputs_piano])
        inputs = inputs.unsqueeze(1)

        domain_y = torch.cat([torch.ones(inputs_ml.shape[0]),
                              torch.zeros(inputs_piano.shape[0])])
        domain_y = domain_y.to(device)

        optimizer.zero_grad()
        output = model(inputs, num_labeled=inputs_ml.shape[0], lambda_=lambda_)
        label_preds = output['output']
        domain_preds = output['domain']
        # emb = output['embedding']

        ml_loss = criterion(label_preds.float(), labels.float())
        domain_loss = F.binary_cross_entropy_with_logits(domain_preds.squeeze(), domain_y)
        loss = ml_loss + domain_loss

        ml_loss_list.append(ml_loss.item())
        domain_loss_list.append(domain_loss.item())
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    writer.add_scalar(f"{kwargs.get('phase', 'training')}/lambda", lambda_, epoch)
    writer.add_scalar(f"{kwargs.get('phase', 'training')}/ml_loss", np.mean(ml_loss_list), epoch)
    writer.add_scalar(f"{kwargs.get('phase', 'training')}/domain_loss", np.mean(domain_loss_list), epoch)

    epoch_loss = np.mean(loss_list)
    writer.add_scalar(f"{kwargs.get('phase', 'training')}/training_loss", epoch_loss, epoch)

    return {'avg_loss': epoch_loss}


def test(model, dataloader, criterion, writer, epoch=-1, **kwargs):
    if kwargs.get('mets') is None:
        mets = ['corr_avg']
    else:
        mets = kwargs['mets']

    model.eval()
    loss_list = []
    preds_list = []
    labels_list = []
    return_dict = {}

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc="Testing ... "):
        song_ids, inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.ndim < 4:
            inputs = inputs.unsqueeze(len(inputs.shape) - 2)
        output = model(inputs)
        try:
            loss = criterion(output['output'].float(), labels.float())
            preds_list.append(output['output'].cpu().detach().numpy())
        except:
            loss = criterion(output.float(), labels.float())
            preds_list.append(output.cpu().detach().numpy())

        loss_list.append(loss.item())
        labels_list.append(labels.cpu().detach().numpy())

    epoch_test_loss = np.mean(loss_list)
    return_dict['avg_loss'] = epoch_test_loss
    if kwargs.get('compute_metrics', True) is True:
        return_dict.update(compute_metrics(np.vstack(labels_list), np.vstack(preds_list), metrics_list=mets))

    if epoch >= 0:
        for k, v in return_dict.items():
            if isinstance(v, float):
                writer.add_scalar(f"test/{k}", v, epoch)

    return return_dict, np.vstack(labels_list), np.vstack(preds_list)


def predict(model, dataloader, mls):
    model.eval()
    preds_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc='Predicting...'):
        song_ids, inputs = batch
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(len(inputs.shape) - 2)

        output = model(inputs)

        ml_preds = output['output'].cpu().detach().numpy()
        preds_list.append(np.hstack([np.array(song_ids).reshape((len(song_ids), 1)), ml_preds]))

    preds_np = np.vstack(preds_list)
    preds = pd.DataFrame(preds_np, columns=['path'] + mls)

    return preds


def predict_multi_model(models, dataloader, mls, aggregate='random'):
    logger.info(f"Multi model predict with {len(models)} models and {aggregate} aggregate...")
    for model in models:
        model.eval()
    preds_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc='Predicting...'):
        song_ids, inputs = batch
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(len(inputs.shape) - 2)

        if aggregate == 'random':
            model = np.random.choice(models)
            output = model(inputs)['output']

        elif aggregate == 'average':
            outs = []
            for model in models:
                outs.append(model(inputs)['output'])
            output = torch.mean(torch.stack(outs, dim=0), dim=0)
        else:
            raise Exception(f"aggregate must be in ['random, 'average'] - is {aggregate}")

        ml_preds = output.cpu().detach().numpy()
        preds_list.append(np.hstack([np.array(song_ids).reshape((len(song_ids), 1)), ml_preds]))

    preds_np = np.vstack(preds_list)
    preds = pd.DataFrame(preds_np, columns=['path'] + mls)

    return preds

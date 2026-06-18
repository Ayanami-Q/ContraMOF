import os
import csv
import yaml
import shutil
import argparse
import sys
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerPretrain, regressoionHead
from model.utils import *
from dataset.dataset_finetune_transformer import MOF_ID_Dataset

warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_ft_transformer.yaml', os.path.join(model_checkpoints_folder, 'config_ft_transformer.yaml'))


class FT_TransformerRegressor(nn.Module):
    def __init__(self, transformer, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.transformer = transformer
        self.regressionHead = regressoionHead(d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        cls_embed = self.transformer(src, return_cls=True)
        output = self.regressionHead(cls_embed)
        return output


class FineTune(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=log_dir)

        self.random_seed = self.config['dataloader']['randomSeed']

        with open(self.config['dataset']['dataPath']) as f:
            reader = csv.reader(f)
            self.mofdata = [row for row in reader]
        self.mofdata = np.array(self.mofdata)
        self.vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(self.vocab_path, model_max_length=512, padding_side='right')

        self.train_data, self.valid_data, self.test_data = split_data(
            self.mofdata, valid_ratio=self.config['dataloader']['valid_ratio'], test_ratio=self.config['dataloader']['test_ratio'],
            randomSeed=self.config['dataloader']['randomSeed']
        )

        self.train_dataset = MOF_ID_Dataset(data=self.train_data, tokenizer=self.tokenizer)
        self.valid_dataset = MOF_ID_Dataset(data=self.valid_data, tokenizer=self.tokenizer)
        self.test_dataset = MOF_ID_Dataset(data=self.test_data, tokenizer=self.tokenizer)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], drop_last=False,
            shuffle=True, pin_memory=False
        )

        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], drop_last=False,
            shuffle=False, pin_memory=False
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], drop_last=False,
            shuffle=False, pin_memory=False
        )

        self.criterion = nn.MSELoss()
        self.normalizer = Normalizer(torch.from_numpy(self.train_dataset.label))

    def _get_device(self):
        if torch.cuda.is_available() and self.config.get('gpu', 'cpu') != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)
        return device

    def train(self):
        self.transformer = TransformerPretrain(
            ntoken=self.config['Transformer']['ntoken'],
            d_model=self.config['Transformer']['d_model'],
            nhead=self.config['Transformer']['nhead'],
            d_hid=self.config['Transformer']['d_hid'],
            nlayers=self.config['Transformer']['nlayers'],
            dropout=self.config['Transformer']['dropout'],
            proj_dim=128,
            pad_token_id=0
        )

        if self.config['cuda']:
            self.transformer = self.transformer.to(self.device)

        model_transformer = self._load_pre_trained_weights(self.transformer)
        model = FT_TransformerRegressor(transformer=model_transformer, d_model=self.config['Transformer']['d_model'])

        if self.config['cuda']:
            model = model.to(self.device)

        # Collect regression head params for higher learning rate
        layer_list = []
        for name, param in model.named_parameters():
            if 'regressionHead' in name:
                print(name, 'new layer (requires higher lr)')
                layer_list.append(name)

        params = [p for n, p in model.named_parameters() if n in layer_list]
        base_params = [p for n, p in model.named_parameters() if n not in layer_list]

        if self.config['optim']['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                [{'params': base_params, 'lr': self.config['optim']['lr'] * 0.2}, {'params': params}],
                self.config['optim']['init_lr'], momentum=self.config['optim'].get('momentum', 0.9),
                weight_decay=eval(str(self.config['optim']['weight_decay']))
            )
        elif self.config['optim']['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                [{'params': base_params, 'lr': self.config['optim']['init_lr'] * 1}, {'params': params}],
                self.config['optim']['init_lr'] * 200, weight_decay=eval(str(self.config['optim']['weight_decay']))
            )
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_mae = np.inf

        model.train()

        for epoch_counter in range(self.config['epochs']):
            for bn, (inputs, target) in enumerate(self.train_loader):
                input_var = inputs.to(self.device, non_blocking=True) if self.config['cuda'] else inputs.to(self.device)

                target_normed = self.normalizer.norm(target)
                target_var = Variable(target_normed.to(self.device, non_blocking=True)) if self.config['cuda'] else Variable(target_normed)

                output = model(input_var)
                loss = self.criterion(output.view(-1), target_var.view(-1))

                if bn % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    print(f'Epoch: {epoch_counter + 1}, Batch: {bn}, Loss: {loss.item():.4f}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss, valid_mae = self._validate(model, self.valid_loader, epoch_counter)
                if valid_mae < best_valid_mae:
                    best_valid_mae = valid_mae
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

        self.model = model

    def _load_pre_trained_weights(self, model):
        fine_tune_path = self.config.get('fine_tune_from', 'scratch')

        if str(fine_tune_path).lower() == 'scratch':
            print("Instruction: 'scratch' detected. Training Transformer from scratch.")
            return model

        checkpoint_path = os.path.join(fine_tune_path, 'model_t_best.pth')

        try:
            load_state = torch.load(checkpoint_path, map_location=self.device)
            model_state = model.state_dict()

            for name, param in load_state.items():
                if name not in model_state:
                    print('NOT loaded (not used in fine-tuning):', name)
                    continue
                else:
                    print('loaded:', name)
                if isinstance(param, nn.parameter.Parameter):
                    param = param.data
                model_state[name].copy_(param)

            print(f"Loaded pre-trained model from {checkpoint_path} with success.")
        except FileNotFoundError:
            print(f"WARNING: Pre-trained weights not found at {checkpoint_path}.")
            print("Falling back to training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_epoch):
        losses = AverageMeter()
        mae_errors = AverageMeter()

        with torch.no_grad():
            model.eval()
            for bn, (inputs, target) in enumerate(valid_loader):
                input_var = inputs.to(self.device, non_blocking=True) if self.config['cuda'] else inputs.to(self.device)
                target_normed = self.normalizer.norm(target)
                target_var = Variable(target_normed.to(self.device, non_blocking=True)) if self.config['cuda'] else Variable(target_normed)

                output = model(input_var)
                loss = self.criterion(output.view(-1), target_var.view(-1))

                mae_error = mae(self.normalizer.denorm(output.data.cpu()).view(-1), target.view(-1))
                losses.update(loss.item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

            print('Epoch [{}] Validate: [{}/{}], Loss {:.4f} ({:.4f}), MAE {:.3f} ({:.3f})'.format(
                n_epoch + 1, bn + 1, len(self.valid_loader), losses.val, losses.avg, mae_errors.val, mae_errors.avg))

        model.train()
        return losses.avg, mae_errors.avg

    def test(self):
        print('Test on test set')
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        losses = AverageMeter()
        mae_errors = AverageMeter()
        test_targets = []
        test_preds = []

        with torch.no_grad():
            self.model.eval()
            for bn, (inputs, target) in enumerate(self.test_loader):
                input_var = inputs.to(self.device)
                target_normed = self.normalizer.norm(target)
                target_var = Variable(target_normed.to(self.device, non_blocking=True)) if self.config['cuda'] else Variable(target_normed)

                output = self.model(input_var)
                loss = self.criterion(output.view(-1), target_var.view(-1))

                mae_error = mae(self.normalizer.denorm(output.data.cpu()).view(-1), target.view(-1))
                losses.update(loss.item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))

                test_pred = self.normalizer.denorm(output.data.cpu())
                test_preds += test_pred.view(-1).tolist()
                test_targets += target.view(-1).tolist()

            print('Test Final: Loss {:.4f} ({:.4f}), MAE {:.3f} ({:.3f})'.format(
                losses.val, losses.avg, mae_errors.val, mae_errors.avg))

        with open(os.path.join(self.writer.log_dir, 'test_results.csv'), 'w') as f:
            writer = csv.writer(f)
            for target, pred in zip(test_targets, test_preds):
                writer.writerow((target, pred))

        self.model.train()
        return losses.avg, mae_errors.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer finetuning')
    parser.add_argument('--seed', default=1, type=int, metavar='Seed', help='random seed for splitting data (default: 1)')
    args = parser.parse_args(sys.argv[1:])

    config = yaml.load(open("config_ft_transformer.yaml", "r"), Loader=yaml.FullLoader)
    config['dataloader']['randomSeed'] = args.seed

    if 'hMOF' in config['dataset']['data_name']:
        task_name = config['dataset']['data_name']
    if 'QMOF' in config['dataset']['data_name']:
        task_name = 'QMOF'

    ftf_val = str(config.get('fine_tune_from', 'scratch'))
    if ftf_val.lower() == 'scratch':
        ptw = 'scratch'
    else:
        ptw = config.get('trained_with', ftf_val.rstrip('/').split('/')[-1])

    seed = config['dataloader']['randomSeed']
    log_dir = os.path.join(
        'training_results/finetuning/Transformer',
        'Trans_{}_{}_{}'.format(ptw, task_name, seed)
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fine_tune = FineTune(config, log_dir)
    fine_tune.train()
    loss, metric = fine_tune.test()

    fn = 'Trans_{}_{}_{}.csv'.format(ptw, task_name, seed)
    df = pd.DataFrame([[loss, metric.item() if hasattr(metric, 'item') else metric]])
    df.to_csv(os.path.join(log_dir, fn), mode='a', index=False, header=False)

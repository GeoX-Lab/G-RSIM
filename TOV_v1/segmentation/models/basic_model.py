# -*- coding:utf-8 -*-
'''
Basic pytorch lightning model for segmentation.
'''
from typing import List
import torch
import torch.nn as nn
# from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import torchmetrics as TM
import pytorch_lightning as pl
from qjdltools.dltrain import SegScore


class BasicModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        freeze_layers: List = [],
        loss: str = 'ce',
        max_epochs: int = 100,
        optimizer: str = 'adam',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        lr_scheduler: str = 'cosine',
        lr_decay_steps: List = [60, 80],
        lr_decay_rate: float = 0.1,
        final_lr: float = 0.,
        nesterov: bool = False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore='model')

        if freeze_layers:
            if 'ExcludeFC' in freeze_layers:
                for name, param in model.named_parameters():
                    if not any(layer in name for layer in ['classifier']):
                        param.requires_grad = False

                if hasattr(model, 'classifier'):
                    model.classifier.weight.data.normal_(mean=0.0, std=0.01)
                    model.classifier.bias.data.zero_()
                print('Kaiming Initialization of classifier\n')
            else:
                for name, param in model.named_parameters():
                    if any(layer in name for layer in freeze_layers):
                        param.requires_grad = False

        self.learning_rate = learning_rate
        self.model = model
        self.configure_loss()

        # metrics | Overall accuracy
        self.num_classes = kwargs.get('num_classes', None)
        self.ignore_index = kwargs.get('ignore_index', None)

        self.train_acc = TM.Accuracy(num_classes=self.num_classes, ignore_index=self.ignore_index)
        self.train_iou = TM.JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index)

        tm_kwargs = {'num_classes': self.num_classes, 'compute_on_step': False}
        self.val_acc = TM.Accuracy(**tm_kwargs, ignore_index=self.ignore_index)
        self.val_iou = TM.JaccardIndex(**tm_kwargs, ignore_index=self.ignore_index)
        self.val_scores = SegScore(self.num_classes, kwargs.get('classes', None))

        self.test_metrics = TM.MetricCollection([
            TM.Accuracy(**tm_kwargs, ignore_index=self.ignore_index),
            # TM.Precision(**tm_kwargs, average='macro', ignore_index=self.ignore_index),
            # TM.Recall(**tm_kwargs, average='macro', ignore_index=self.ignore_index),
            TM.JaccardIndex(**tm_kwargs, ignore_index=self.ignore_index),
            # TM.F1Score(**tm_kwargs, ignore_index=self.ignore_index),
            # TM.ConfusionMatrix(**tm_kwargs)
        ])

        self.val_verbose = False
        self.final_val = False

    def on_train_epoch_start(self) -> None:
        if self.hparams.freeze_layers:
            if self.current_epoch == 0:
                print('Freeze layer:')
            for layer in self.hparams.freeze_layers:
                if hasattr(self.model, layer):
                    getattr(self.model, layer).eval()
                    if self.current_epoch == 0:
                        print(f'\t{layer}')

    def shared_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['label'].long()

        logits = self.model(x)['out']

        return logits, y

    def training_step(self, batch, batch_idx):
        preds, y = self.shared_step(batch, batch_idx)
        loss = self.loss_function(preds, y)

        return {'loss': loss, 'preds': preds, 'target': y}

    def training_step_end(self, outputs):
        preds = outputs['preds'].argmax(dim=1)
        self.train_acc(preds, outputs['target'])
        self.train_iou(preds, outputs['target'])

        self.log('train/loss', outputs['loss'], prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/mIoU', self.train_iou, on_step=True, on_epoch=True, prog_bar=True)

    # def training_epoch_end(self, outputs):
    #     self.val_scores.reset()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            preds, y = self.shared_step(batch, batch_idx)
            return {'preds': preds, 'target': y}

    def validation_step_end(self, outputs):
        preds = outputs['preds'].argmax(dim=1)
        self.val_acc.update(preds, outputs['target'])
        self.val_iou.update(preds, outputs['target'])

        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/mIoU', self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

        self.val_scores.update(
            outputs['target'].cpu().numpy(), preds.cpu().numpy())

    def validation_epoch_end(self, outputs):
        scores = self.val_scores.get_scores()
        if self.val_verbose:
            print('\n\nval OA={:.2f} | AA={:.2f} | Mean IoU={:.2f}'.format(
                scores['OA'], scores['AA'], scores['Mean IoU']))

        self.log('val/OA', scores['OA'])
        self.log('val/AA', scores['AA'])
        # self.log('val/Mean_IoU', scores['Mean IoU'])
        self.log('val/Mean_IoU_by_TM', self.val_iou)

        if self.final_val:
            self.val_scores.print_score(scores)
            self.val_scores.print_hist(scores['hist'])

        self.epoch_scores = scores
        self.val_scores.reset()

    def on_test_epoch_start(self) -> None:
        # num_classes = self.hparams.get('num_classes', None)
        pass

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        with torch.no_grad():
            preds, y = self.shared_step(batch, batch_idx)
        preds = preds.argmax(dim=1)
        self.test_metrics.update(preds, y)

    # def test_step_end(self, outputs):

    def test_epoch_end(self, outputs):
        self.test_scores = self.test_metrics.compute()

    def configure_optimizers(self):
        params = self.parameters()
        params = list(filter(lambda p: p.requires_grad, params))

        weight_decay = self.hparams.get('weight_decay', 0)

        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                params, lr=self.learning_rate,
                weight_decay=weight_decay)
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                params, lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
                nesterov=self.hparams.nesterov)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.MultiStepLR(
                    optimizer, self.hparams.lr_decay_steps,
                    gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(
                    optimizer, T_max=self.hparams.max_epochs,
                    eta_min=self.hparams.final_lr)
            elif self.args.scheduler == "cosineanneal":
                scheduler = lrs.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10, T_mult=1, eta_min=1e-5,
                    last_epoch=-1)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()

        if loss == 'ce':
            criterion = nn.CrossEntropyLoss
        elif loss == 'bce':
            criterion = nn.BCEWithLogitsLoss
        elif loss == 'mse':
            criterion = nn.MSELoss
        elif loss == 'l1':
            criterion = nn.L1Loss
        elif loss == 'mlsm':
            criterion = nn.MultiLabelSoftMarginLoss
        else:
            raise ValueError("Invalid Loss Type!")

        self.loss_function = criterion()

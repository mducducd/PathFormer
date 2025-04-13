from typing import Optional, Union, Sequence, Dict, Literal, Any
import models as module_arch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Identity, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics 

class Classifier(LightningModule):

    def __init__(self, cfg: Dict,
        task: Literal["binary", "multiclass", "multilabel"] = "binary",
        distributed: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.n_classes = cfg.General.n_classes
        self.learning_rate = cfg.Optimizer.lr
        self.distributed = distributed
        self.model_name = cfg.Model.name
   
        # if self.n_classes > 2: 
        self.loss_fn = CrossEntropyLoss()
        self.auc_fn = torchmetrics.AUROC(task='multiclass', num_classes = self.n_classes, average = 'macro')
        self.acc_fn = torchmetrics.Accuracy(task='multiclass', num_classes= self.n_classes)
            # metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='multiclass', num_classes = self.n_classes,
            #                                                                average='micro'),
            #                                          torchmetrics.CohenKappa(task='multiclass', num_classes = self.n_classes),
            #                                          torchmetrics.F1Score(task='multiclass', num_classes = self.n_classes,
            #                                                          average = 'macro'),
            #                                          torchmetrics.Recall(task='multiclass', average = 'macro',
            #                                                              num_classes = self.n_classes),
            #                                          torchmetrics.Precision(task='multiclass', average = 'macro',
            #                                                                 num_classes = self.n_classes),
            #                                          torchmetrics.Specificity(task='multiclass', average = 'macro',
            #                                                                 num_classes = self.n_classes)])
        # else : 
        #     self.loss_fn = CrossEntropyLoss()
        #     self.auc_fn = torchmetrics.AUROC(task='binary', num_classes=2, average = 'macro')
        #     self.acc_fn = torchmetrics.Accuracy(task='binary', num_classes=num_classes)
        #     # metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary', num_classes = 2,
        #     #                                                                average = 'micro'),
        #     #                                          torchmetrics.CohenKappa(task='binary', num_classes = 2),
        #     #                                          torchmetrics.F1Score(task='binary', num_classes = 2,
        #     #                                                          average = 'macro'),
        #     #                                          torchmetrics.Recall(task='binary', average = 'macro',
        #     #                                                              num_classes = 2),
        #     #                                          torchmetrics.Precision(task='binary', average = 'macro',
        #     #                                                                 num_classes = 2)])
            

        # if finetune:
        #     pass
        # else:
        self.model = getattr(module_arch, self.model_name)(**cfg.Model.args)


    @classmethod
    def from_module(cls, model, learning_rate: float = 1e-4, distributed=False):
        return cls(model, learning_rate, distributed)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]]) -> Dict[str, Tensor]:
        if self.model_name == 'VisionTransformer':
            bags, y, coords, bagSizes = batch
            y_hat = self(bags, coords=coords)
            y_hat = y_hat
        else:
            x, y, *rest = batch
            y_hat = self(x)
            y_hat = y_hat

        # if self.task == "multilabel":
        #     y_hat = y_hat.flatten()
        #     y = y.flatten()
        loss = self.loss_fn(y_hat, y)

        prob = y_hat.sigmoid()
        acc = self.acc_fn(prob, y)
        auc = self.auc_fn(prob, y)
        return {"loss": loss, "acc": acc, "auc": auc}
    
    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        loss_dict = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        loss_dict = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        return loss_dict["loss"]

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self(batch[0])

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=7, verbose=True, min_lr=1e-8),
                "monitor": "train_loss"
            }
        }
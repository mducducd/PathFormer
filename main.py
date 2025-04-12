import argparse
from datasets import WSIDatasetModule
from models import Classifier
from utils.utils import *

# pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm
from utils.earlystop_lr import EarlyStoppingLR
from utils.lr_logger import LrLogger
from utils.seed import Seed
from utils.system_stats_logger import SystemStatsLogger

# Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='config\config.yaml',type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--gpus', default = [1])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to evaluate.")
    parser.add_argument("--skip_train", action="store_true", default=False, help="Skip training and evaluate only.")
    args = parser.parse_args()
    return args

def train(args, cfg):
    resume_ckpt = args.resume
    max_epochs = cfg.General.epochs
    n_gpus = 1

    # Define Model 
    model = Classifier(
            num_classes=cfg.Model.n_classes, task="binary", learning_rate=cfg.Optimizer.lr, model_name='TransMIL'
        )
    
    # Define Data 
    dm = WSIDatasetModule(
            data_dir=cfg.Data.data_dir,
            label_dir=cfg.Data.label_dir,
            batch_size=args.batch_size,
            num_workers=1, 
    )

    if args.skip_train:
        dm.setup()
        return resume_ckpt, dm
    
    # strategy = None 
    # accelerator =  "gpu"

    ckpt_filename = cfg.Model.name + "-{epoch}-{val_auc:.3f}"
    ckpt_monitor = "val_auc"

    ckpt_callback = ModelCheckpoint(dirpath=f"ckpt/{cfg.Model.name}", save_last=True,
            filename=ckpt_filename,
            monitor=ckpt_monitor,
            mode="max")
    
    trainer = Trainer(log_every_n_steps=1, devices=[0], accelerator='gpu', benchmark=True,
            logger=True, precision=16, max_epochs=max_epochs,
            strategy=None, resume_from_checkpoint=resume_ckpt,
            callbacks=[ckpt_callback, LrLogger(), EarlyStoppingLR(1e-6), SystemStatsLogger()])
    
    trainer.fit(model, dm)
    # trainer.test(model, datamodule=dm)

    return ckpt_callback.best_model_path, dm

def evaluate_celebvhq(args, cfg, ckpt):
    print("Load checkpoint", ckpt)
    model = Classifier.load_from_checkpoint(ckpt)
    trainer = Trainer(log_every_n_steps=1, devices=[0], accelerator="gpu", benchmark=True,
        logger=False, enable_checkpointing=False)
    Seed.set(42)
    model.eval()

    dm = WSIDatasetModule(
        data_dir=cfg.Data.data_dir,
        label_dir=cfg.Data.label_dir,
        batch_size=1,  # During test samples are vary in N titles
        num_workers=1, 
    )
    dm.setup()

    # collect predictions
    preds = trainer.predict(model, dm.test_dataloader())
    preds = torch.cat(preds)
    
    # collect ground truth
    preds = torch.argmax(preds.sigmoid(), dim=1)
    ys = torch.zeros_like(preds, dtype=torch.bool)
 
    for i, (_, y) in enumerate(tqdm(dm.test_dataloader())):
        ys[i * args.batch_size: (i + 1) * args.batch_size] = y

    # Eval
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve
  
    # Convert one-hot encoded labels to class indices
    y_true = ys.numpy()
    # y_pred = preds_bool.numpy()
    y_pred = preds.numpy()

    # from torcheval.metrics.functional import multiclass_f1_score
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1_scores = f1_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)


    print(accuracy, precision, recall, f1_scores)


def main(args):
    cfg = read_yaml(args.config)

    # update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage

    if args.stage == 'train':
        ckpt, dm = train(args, cfg)
    else:
        evaluate_celebvhq(args, cfg, args.ckpt)
   
if __name__ == '__main__':
    args = make_parse()

    main(args)
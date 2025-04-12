import inspect # 查看python 类的参数和模块、函数代码
import importlib # In order to dynamically import the library
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
from pathlib import Path
import torch
import random
from typing import Optional

class WSIDataset(LightningDataModule, ABC):
    def __init__(self, sampleid_to_path: list, label_df: pd.DataFrame, state: str):
        super().__init__()
   
        self.label_df = label_df
        self.state = state

        self.data = label_df['sampleId'].tolist()
        self.label = label_df['label'].tolist()
        

        # Match sample_id from filename
        self.sampleid_to_path = sampleid_to_path
   

    def __getitem__(self, idx: int):
        sample_id = self.data[idx]
        label = self.label[idx]
        
        # Load .h5 feature file
        h5_path = self.sampleid_to_path[sample_id]

        with h5py.File(h5_path, 'r') as f:
            augmented = torch.tensor(f['augmented'][:])
            coords = torch.tensor(f['coords'][:], dtype=torch.float)
            feats = torch.tensor(f['feats'][:], dtype=torch.float)
        
        if self.state == 'train' or self.state == 'val':
            window_size = 128
            num_tiles = feats.shape[0]

            if num_tiles < window_size:
                raise ValueError(f"Sample '{sample_id}' has only {num_tiles} tiles, which is less than window size {window_size}")

            # Random slide
            cp = random.randint(0, num_tiles - window_size)
            feats = feats[cp:cp + window_size]
            coords = coords[cp:cp + window_size]
            augmented = augmented[cp:cp + window_size]

        # Shuffle features if needed
        # if self.shuffle:
        #     indices = list(range(feats.shape[0]))
        #     random.shuffle(indices)
        #     feats = feats[indices]
        #     coords = coords[indices]
        #     augmented = augmented[indices]

        # return sample_id, feats, label, coords, augmented, label
        return feats, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

class WSIDatasetModule(LightningDataModule):
    def __init__(self, 
                data_dir: str,
                label_dir: str,
                batch_size=16, 
                num_workers=1, 
                take_train: Optional[int] = None,
                take_val: Optional[int] = None,
                take_test: Optional[int] = None):
        super().__init__()
        
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.take_train = take_train
        self.take_val = take_val
        self.take_test = take_test
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        #----> read .txt file and process it
        raw_df = pd.read_csv(self.label_dir, sep='\t')
        raw_df.columns = [col.strip() for col in raw_df.columns]

        #----> extract sampleId from studyID:sampleId
        raw_df['sampleId'] = raw_df.iloc[:, 0].apply(lambda x: x.split(':')[1])
        raw_df['label'] = raw_df['Altered'].astype(int)

        #----> find valid .h5 files
        available_h5_files = list(Path(self.data_dir).glob("*.h5"))

        ### Some samples in .txt dont have coressponding .h5 
        valid_sample_ids = set()
        self.sampleid_to_path = {}
        for file in available_h5_files:
            base = file.stem
            if 'Z' in base:
                sid = base.split('Z-')[0]
                valid_sample_ids.add(sid)
                self.sampleid_to_path[sid] = file # Match sample_id from filename


        #----> filter raw_df to keep only rows with valid sampleIds
        raw_df = raw_df[raw_df['sampleId'].isin(valid_sample_ids)].reset_index(drop=True)

        #----> split data
        train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42, stratify=raw_df['label'])
        # train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label'])

        #----> assign datasets
        self.train_dataset = WSIDataset(self.sampleid_to_path, train_df, "train")
        self.val_dataset = WSIDataset(self.sampleid_to_path, test_df, "val")
        self.test_dataset = WSIDataset(self.sampleid_to_path, test_df, "test")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

class DataInterface(LightningDataModule):

    def __init__(self, train_batch_size=64, train_num_workers=8, test_batch_size=1, test_num_workers=1,dataset_name=None, **kwargs):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """        
        super().__init__()

        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        self.load_data_module()

 

    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """  
        - count number of classes

        - build vocabulary

        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.instancialize(state='train')
            self.val_dataset = self.instancialize(state='val')
 

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.test_dataset = self.instancialize(state='test')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False)


    def load_data_module(self):
        camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])

        try:
            self.data_module = getattr(importlib.import_module(
                f'datasets.{self.dataset_name}'), camel_name)
        except:
            raise ValueError(
                'Invalid Dataset File Name or Invalid Class Name!')
    
    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = list(inspect.signature(self.data_module.__init__).parameters.keys())[1:]
        print('ffffffffffffffff', class_args)
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
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
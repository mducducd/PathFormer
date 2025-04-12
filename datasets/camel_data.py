import torch
from torch.utils import data
import pandas as pd
import random
from pathlib import Path
import h5py
from sklearn.model_selection import train_test_split

class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None, state=None):
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #----> read .txt file and process it
        raw_df = pd.read_csv(self.dataset_cfg.label_dir, sep='\t')
        raw_df.columns = [col.strip() for col in raw_df.columns]

        #----> extract sampleId from studyID:sampleId
        raw_df['sampleId'] = raw_df.iloc[:, 0].apply(lambda x: x.split(':')[1])
        raw_df['label'] = raw_df['Altered'].astype(int)  # or CDH1 if you prefer

        #----> split data
        train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42, stratify=raw_df['label'])
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label'])

        if state == 'train':
            self.data = train_df['sampleId'].tolist()
            self.label = train_df['label'].tolist()
        elif state == 'val':
            self.data = val_df['sampleId'].tolist()
            self.label = val_df['label'].tolist()
        elif state == 'test':
            self.data = test_df['sampleId'].tolist()
            self.label = test_df['label'].tolist()
        else:
            raise ValueError("State must be 'train', 'val', or 'test'.")

        self.feature_dir = self.dataset_cfg.data_dir
        self.shuffle = self.dataset_cfg.data_shuffle

        # ---> Preload all .h5 paths
        self.feature_dir = Path(self.dataset_cfg.data_dir) 
        all_h5_files = list(self.feature_dir.glob("*.h5"))
        self.sampleid_to_path = {
            path.name.split('Z')[0]: path
            for path in all_h5_files
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data[idx]
        label = self.label[idx]

        # Load .h5 feature file
        h5_path = self.sampleid_to_path[sample_id]

        with h5py.File(h5_path, 'r') as f:
            augmented = torch.tensor(f['augmented'][:])
            coords = torch.tensor(f['coords'][:])
            feats = torch.tensor(f['feats'][:])

        # Shuffle features if needed
        if self.shuffle:
            indices = list(range(feats.shape[0]))
            random.shuffle(indices)
            feats = feats[indices]
            coords = coords[indices]
            augmented = augmented[indices]

        # return sample_id, feats, label, coords, augmented, label
        return feats, label
    

# from torch.utils.data import DataLoader
# from types import SimpleNamespace
# cfg = SimpleNamespace(
#         label_dir='D:\\task_dresen\\sample_matrix.txt',
#         data_dir='D:\\task_dresen\\TCGA_BRCA',
#         data_shuffle=True,
#         test_size=0.2,
#         val_size=0.1
#     )
# dataset = CamelData(dataset_cfg = cfg, state='train')
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# # Get one sample
# sample = next(iter(dataloader))
# sample_id, feats, label, coords, augmented, _ = sample
# print(sample_id, feats.shape, label, coords.shape, augmented.shape)


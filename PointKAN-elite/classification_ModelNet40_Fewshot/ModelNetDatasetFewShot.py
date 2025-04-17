import os
import numpy as np
import pickle
from torch.utils.data import Dataset
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data/ModelNetFewshot')

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ModelNet40FewShot(Dataset):
    def __init__(self, num_points, partition='train', way=10, shot=20, prefix_ind=5):
        self.num_points = num_points
        self.partition = partition
        save_path = os.path.join(DATA_DIR, f'{way}way_{shot}shot')
        with open(os.path.join(save_path, f'{prefix_ind}.pkl'), 'rb') as f:
            self.data = pickle.load(f)[self.partition]
        
    def __getitem__(self, item):
        pointcloud, label, _ = self.data[item]
        pointcloud = pointcloud[:self.num_points]
        pointcloud = pointcloud[:, 0:3]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return len(self.data)
    
if __name__ == '__main__':
    train = ModelNet40FewShot(1024)
    test = ModelNet40FewShot(1024, 'test')
    for data, label in test:
        print(data.shape)
        # print(label.shape)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(ModelNet40FewShot(partition='train', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")

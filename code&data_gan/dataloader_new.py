from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torchvision.transforms as transforms
import pandas as pd
import os

result = pd.read_csv("result.csv")
path_before = "split_dataset/before"
path_after = "split_dataset/after"

class MultiImageDataset(Dataset):
    def __init__(self, id, features, indices, city, labels, imagedata_before, imagedata_after):
        super().__init__()
        self.id = id
        self.features = features
        self.indices = indices
        self.labels = labels
        self.city = city
        self.imagedata_before = imagedata_before
        self.imagedata_after = imagedata_after

    def __getitem__(self, index):
        return self.id[index], self.features[index], self.indices[index], self.labels[index], self.imagedata_after[
            index], self.imagedata_before[index]

    def __len__(self):
        return len(self.imagedata_before)


# path test/train/val
def load_data(path: str, batch_size: int = 2, shuffle: bool = False, input_shape: int = 512):
    std_scaler = StandardScaler()
    labelencoder = LabelEncoder()
    transform = transforms.Compose([
        transforms.Resize(input_shape),
        transforms.ToTensor()
    ])

    path_before_ = path_before + '/' + path
    path_after_ = path_after + '/' + path

    imagedata_before = ImageFolder(root = path_before_, transform=transform)
    imagedata_after = ImageFolder(root = path_after_, transform=transform)

    
    imgname_a = [os.path.basename(img[0]) for img in imagedata_after.imgs]
    imgid_a = [int(name[:-4]) for name in imgname_a]

    dataset_id = imgid_a
    selected_res = result.loc[dataset_id]
    selected_features = selected_res.iloc[:, 2:29]
    selected_features = std_scaler.fit_transform(selected_features)
    selected_indices = selected_res.iloc[:, 29:32]
    selected_indices = std_scaler.fit_transform(selected_indices)

    labels = [img[1] for img in imagedata_after.imgs]
    cities = selected_res["city"]

    dataset = MultiImageDataset(id = dataset_id, features = selected_features, indices = selected_indices, city = cities,
                                labels = labels, imagedata_after = imagedata_after, imagedata_before = imagedata_before)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    print("Finished!")
    return train_loader


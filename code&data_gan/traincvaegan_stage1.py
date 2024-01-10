from model.CVAE_GAN import Classifier
from dataloader_new import load_data
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import torch.nn.parallel as parallel
import torch.optim as optim
import torch.nn as nn
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser('name space')
        parser.add_argument("--path", type = str, default = "train")
        parser.add_argument("--batch_size", type = int, default = 64)
        parser.add_argument("--lr", type = float, default = 1e-3)
        parser.add_argument('--epochs', type = int, default = 50)
        parser.add_argument("--input_shape", type = int, default = 512)
        
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args
    
def train(train_loader, lr: float, epochs: int = 50):
    torch.cuda.set_device("cuda:0")
    C = Classifier().cuda()
    C = parallel.DataParallel(C)
    
    optimizer = optim.Adam(C.parameters(), lr = lr)
    scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = 50, eta_min = 5e-4)
    
    criterion_1 = nn.MSELoss()
    criterion_2 = nn.CrossEntropyLoss()
    
    min_loss = 10000
    
    for epoch in tqdm(range(epochs)):
        Loss = 0.0
        for data in tqdm(train_loader):
            id, features, indices, labels, (imagedata_after, after_label), (imagedata_before, before_label) = data
            features, labels, imagedata_after = features.cuda(), labels.cuda(), imagedata_after.cuda()
            features = features.float()
            features_hat, labels_hat = C(imagedata_after)
            
            loss = criterion_1(features, features_hat) + criterion_2(labels_hat, labels.view(-1))
            
            loss.backward()
            optimizer.step()
            Loss += loss.item()
        
        mean_loss = Loss / len(train_loader)
        if mean_loss < min_loss:
            torch.save(C.state_dict(), f"classifier{args.input_shape}_best.pth")
            min_loss = mean_loss
        
        print(f"Epoch {epoch + 1}:")
        print(f"Classifier loss: {mean_loss:.4f}")
        
        scheduler.step()
        
if __name__ == "__main__":
    args = Options().parse()
    train_loader = load_data(path = args.path,
                             batch_size = args.batch_size,
                             shuffle = True,
                             input_shape = args.input_shape)
    
    train(train_loader, lr = args.lr, epochs = args.epochs)
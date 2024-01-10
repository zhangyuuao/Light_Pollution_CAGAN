from model.CVAE_GAN import CVAE_GAN, Discriminator, Classifier
from model.Loss import ELBO_Loss
from dataloader_new import load_data
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import argparse
import torch
import numpy as np
import torch.nn.parallel as parallel
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device("cuda:0")

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser('name space')
        parser.add_argument("--path", type = str, default = "train")
        parser.add_argument("--batch_size", type = int, default = 64)
        parser.add_argument("--lr", type = float, default = 1e-5)
        parser.add_argument("--embedding_dim", type = int, default = 64)
        parser.add_argument('--input_shape', type = int, default = 512)
        parser.add_argument('--kld_weight', type = float, default = 0.2)
        parser.add_argument('--lambda1', type = float, default = 1)
        parser.add_argument('--lambda2', type = float, default = 1e-3)
        parser.add_argument('--is_continue', type = bool, default = False)
        parser.add_argument('--load_model', type = str, default = "cvae_512_50.pth")
        parser.add_argument('--epochs', type = int, default = 50)
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args
    
def save_image(file_path, image):
    image = np.uint8(image * 255.0)
    image = Image.fromarray(image.transpose(1, 2, 0), mode = 'RGB')
    image.save(file_path)

def evaluate(model):
    train_loader2 = load_data(path = "train", batch_size = 1, input_shape = 512)
    for i, data in enumerate(train_loader2):
        with torch.no_grad():
            id, features, indices, labels, (imagedata_after, after_label), (imagedata_before, before_label) = data
            features, labels, imagedata_after = features.cuda(), labels.cuda(), imagedata_after.cuda()
            generated_images, _, _ = model(imagedata_before, features, labels)
            
        generated_images = generated_images.squeeze(0).detach().cpu().numpy()
        generated_images = np.where(generated_images < 0, 0, generated_images)  
        save_image(f"./{args.path}_gen/{id.item()}.jpeg", generated_images)
        if i == 100:
            break
    pass


def train(train_loader, lr: float, embedding_dim: int, epochs: int,
          kld_weight: float, lambda1: float, lambda2: float,
          gamma1: float = 0.9, gamma2: float = 0.9,
          save_interval: int = 10, is_continue: bool = False, load_model: str = "./model_pth/cvae_512_50.pth", load_disc: str = "./model_pth/discriminator_512_50.pth"):
    
    # Initialize and load model
    cvae_gan = CVAE_GAN(embedding_dim = embedding_dim).cuda()
    D = Discriminator().cuda()
    C = Classifier().cuda()
    
    cvae_gan = parallel.DataParallel(cvae_gan)
    D = parallel.DataParallel(D)
    C = parallel.DataParallel(C)
    
    C.load_state_dict(torch.load(f"./model_pth/classifier{args.input_shape}_best.pth"))
    
    if is_continue:
        cvae_gan.load_state_dict(torch.load(load_model))
        D.load_state_dict(torch.load(load_disc))
        print(f"load model: {load_model} successfully!")
        evaluate(cvae_gan)
    
    elbo_loss = ELBO_Loss(kld_weight)
    
    optimizer_G = optim.AdamW(cvae_gan.parameters(), lr = lr)
    optimizer_D = optim.AdamW(D.parameters(), lr = lr)
    
    scheduler_G = CosineAnnealingLR(optimizer = optimizer_G, T_max = 50, eta_min = 1e-5)
    scheduler_D = CosineAnnealingLR(optimizer = optimizer_G, T_max = 50, eta_min = 1e-5)
    
    loss_mse = nn.MSELoss()
    loss_bce = nn.BCELoss()
    loss_crossentropy = nn.CrossEntropyLoss()
    
    # train process
    for epoch in tqdm(range(epochs)):
        Loss = 0
        for data in tqdm(train_loader):
            id, features, indices, labels, (real_images, after_label), (init_images, before_label) = data
            features, labels, init_images, real_images = features.cuda(), labels.cuda(), init_images.cuda(), real_images.cuda()
            batch_size = real_images.size(0)
            
            # train discriminator
            optimizer_D.zero_grad()
            fake_images, mu, log_var = cvae_gan(init_images, features, labels)
            
            _, real_outputs = D(real_images)
            _, fake_outputs = D(fake_images.detach())
            
            real_loss = loss_bce(real_outputs.view(-1), torch.ones(batch_size).cuda())
            fake_loss = loss_bce(fake_outputs.view(-1), torch.zeros(batch_size).cuda())

            disc_loss = (real_loss + fake_loss) / 2
            disc_loss.backward()
            optimizer_D.step()
        
            # train generator
            optimizer_G.zero_grad()
            
            real_feature_D, real_outputs_D = D(real_images)
            fake_feature_D, fake_outputs_D = D(fake_images.detach())
            
            real_feature_C, real_outputs_C = C(real_images)
            fake_feature_C, fake_outputs_C = C(fake_images.detach())
            
            loss_g = loss_mse(real_feature_D, fake_feature_D) + loss_mse(real_feature_C, fake_feature_C)
            loss_gd = loss_mse(torch.mean(real_feature_D, dim = 0), torch.mean(fake_feature_D, dim = 0))
            loss_recon_kl = elbo_loss(fake_images, real_images, mu, log_var)
            loss_d = loss_bce(fake_outputs_D.view(-1), torch.ones(batch_size).cuda())
            loss_c = loss_crossentropy(fake_outputs_C, labels.view(-1))
            
            loss = loss_recon_kl + loss_c + loss_d + lambda1 * loss_g + lambda2 * loss_gd
            
            loss.backward()
            optimizer_G.step()
            Loss += loss.item()
            
        # update lr
        scheduler_G.step()
        scheduler_D.step()
        
        # print loss
        print(f"Epoch {epoch + 1}:")
        print(f"Discriminator loss: {disc_loss.item():.4f}")
        print(f"CVAE loss: {Loss / len(train_loader):.4f}")
        print(f"lr: {scheduler_G.get_last_lr()[0], scheduler_D.get_last_lr()[0]}")
        
        # save model and evalute the images
        if (epoch + 1) % save_interval == 0:
            torch.save(cvae_gan.state_dict(), f"cvae_{args.input_shape}_{epoch + 1}.pth")
            torch.save(D.state_dict(), f"discriminator_{args.input_shape}_{epoch + 1}.pth")
            evaluate(cvae_gan)

if __name__ == "__main__":
    args = Options().parse()
    print(device)
    train_loader = load_data(path = args.path,
                             batch_size = args.batch_size,
                             shuffle = True,
                             input_shape = args.input_shape)
    
    train(train_loader, args.lr, args.embedding_dim, args.epochs, args.kld_weight, args.lambda1, args.lambda2, is_continue = args.is_continue)

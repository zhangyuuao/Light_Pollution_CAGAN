from model.CVAE_GAN import CVAE_GAN
from dataloader_new import load_data
from PIL import Image
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.parallel as parallel


def save_image(file_path, image):
    image = np.uint8(image * 255.0)
    image = Image.fromarray(image.transpose(1, 2, 0), mode = 'RGB')
    image.save(file_path)

def evaluate():
    train_loader2 = load_data(path = "train", batch_size = 1, input_shape = 512)
    cvae_gan = CVAE_GAN().cuda()
    cvae_gan = parallel.DataParallel(cvae_gan)
    cvae_gan.load_state_dict(torch.load("./model_pth/cvae_512_50.pth"))
    for i, data in enumerate(train_loader2):
        with torch.no_grad():
            id, features, indices, labels, (imagedata_after, after_label), (imagedata_before, before_label) = data
            features, labels, imagedata_after = features.cuda(), labels.cuda(), imagedata_after.cuda()
            generated_images, _, _ = cvae_gan(imagedata_before, features, labels)
            
        generated_images = generated_images.squeeze(0).detach().cpu().numpy()
        generated_images = np.where(generated_images < 0, 0, generated_images)  
        save_image(f"./result/{id.item()}.jpeg", generated_images)
        if i == 100:
            break
    pass


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser('name space')
        parser.add_argument("--load_path", type = str, default = "")
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

if __name__ == "__main__":
    args = Options().parse()
    evaluate()
    
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
import torch
import torchvision.models as models

class CVAE(nn.Module):
    def __init__(self,
                 img_size: int = 512,
                 z_dim: int = 100,
                 feature_dim: int = 27,
                 embedding_dim: int = 64,
                 in_channels: int = 3) -> None:
        
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.emb_feature = nn.Linear(feature_dim + embedding_dim, img_size ** 2)
        self.emb_feature1 = nn.Linear(feature_dim + embedding_dim, 256 ** 2)
        self.emb_feature2 = nn.Linear(feature_dim + embedding_dim, 128 ** 2)
        self.emb_feature3 = nn.Linear(feature_dim + embedding_dim, 64 ** 2)
        self.emb_feature4 = nn.Linear(feature_dim + embedding_dim, 32 ** 2)
        self.emb_feature5 = nn.Linear(feature_dim + embedding_dim, 16 ** 2)
        
        self.emb_img = nn.Conv2d(in_channels, in_channels, kernel_size = 1)
        
        # initialize encoder
        self.encoder_1 = self._Convblock(3, 32, 4, 2, 1)
        self.encoder_2 = self._Convblock(32, 64, 4, 2, 1)
        self.encoder_3 = self._Convblock(64, 128, 4, 2, 1)
        self.encoder_4 = self._Convblock(128, 256, 4, 2, 1)
        self.encoder_5 = self._Convblock(256, 512, 4, 2, 1)
        
        self.infer_mu = nn.Linear(512 * 16 * 16, z_dim)
        self.infer_var = nn.Linear(512 * 16 * 16, z_dim)
        
        # initialize decoder
        self.decoder_input = nn.Linear(z_dim + feature_dim + embedding_dim, 512 * 16 * 16)
        
        self.decoder_1 = self._TransConvblock(512, 256, 4, 2, 1)
        self.decoder_2 = self._TransConvblock(256, 128, 4, 2, 1)
        self.decoder_3 = self._TransConvblock(128, 64, 4, 2, 1)
        self.decoder_4 = self._TransConvblock(64, 32, 4, 2, 1)
        self.decoder_5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )
        self.activation = nn.Tanh()
        
    def noise_reparameterize(self, mu, log_var):
        # generate latent vector
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, input, feature):
        
        # input1: 32 256 256
        input1 = self.encoder_1(input)
        feature1 = self.emb_feature1(feature).view(-1, 1, 256, 256)
        input1 = input1 + feature1
        
        # input2: 64 128 128
        input2 = self.encoder_2(input1)
        feature2 = self.emb_feature2(feature).view(-1, 1, 128, 128)
        input2 = input2 + feature2
        
        # input3: 128 64 64
        input3 = self.encoder_3(input2)
        feature3 = self.emb_feature3(feature).view(-1, 1, 64, 64)
        input3 = input3 + feature3
        
        # input4: 256 32 32
        input4 = self.encoder_4(input3)
        feature4 = self.emb_feature4(feature).view(-1, 1, 32, 32)
        input4 = input4 + feature4
        
        # input5: 512 16 16
        input5 = self.encoder_5(input4)
        feature5 = self.emb_feature5(feature).view(-1, 1, 16, 16)
        input5 = input5 + feature5
        
        result = torch.flatten(input5, start_dim = 1)
        mu = self.infer_mu(result)
        log_var = self.infer_var(result)
        
        return input1, input2, input3, input4, input5, mu, log_var
    
    def decode(self, input, input1, input2, input3, input4, input5, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 16, 16)
        result = result + input5
        
        result = self.decoder_1(result)
        result = result + input4
        
        result = self.decoder_2(result)
        result = result + input3
        
        result = self.decoder_3(result)
        result = result + input2
    
        result = self.decoder_4(result)
        result = result + input1
        
        result = self.decoder_5(result)
        result = result + input
        
        return result
    
    def forward(self, x, feature, label_emb):
        feature = torch.cat([feature, label_emb], dim = 1)
        feature = feature.float()

        img_emb = self.emb_img(x)
        feature_emb = self.emb_feature(feature).view(-1, 1, 512, 512)
        img_emb = img_emb + feature_emb
        input1, input2, input3, input4, input5, mu, log_var = self.encode(img_emb, feature)
        
        z = self.noise_reparameterize(mu, log_var)
        z = torch.cat([z, feature], dim = 1)
        
        x_recon = self.decode(img_emb, input1, input2, input3, input4, input5, z)
        x_recon = self.activation(x_recon)
        return x_recon, mu, log_var
    
    def _Convblock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def _TransConvblock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            
    def forward(self, figure):
        features = self.feature_extractor(figure)
        features = torch.flatten(features, 1) 
        return features


class Classifier(nn.Module):
    def __init__(self, label_size: int = 4):
        super().__init__()
        self.backbone = FeatureExtractor(resnet50(weights = ResNet50_Weights.DEFAULT))
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 27),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(27, label_size)
        
        # freeze the parameter of pre-trained model
        for params in self.backbone.parameters():
            params.requires_grad = False
    
    def forward(self, images):
        features = self.backbone(images)
        features = self.fc(features)
        return features, self.classifier(features)

class Discriminator(nn.Module):
    def __init__(self, n_classes = 1):
        super().__init__()
        self.backbone = FeatureExtractor(resnet34(weights = ResNet34_Weights.DEFAULT))
        
        # freeze the parameter of pre-trained model
        for params in self.backbone.parameters():
            params.requires_grad = False
          
        self.fc = nn.Sequential(
            nn.Linear(512, n_classes),
            nn.Sigmoid()
        )

    def forward(self, images):
        feature = self.backbone(images)
        outputs = self.fc(feature)
        return feature, outputs

class CVAE_GAN(nn.Module):
    def __init__(self,
                 embedding_dim = 64,
                 label_size = 4) -> None:
        super().__init__()
        
        self.label_emb = nn.Embedding(label_size, embedding_dim)
        self.cvae = CVAE(embedding_dim = embedding_dim)
    
    def forward(self, x, features, label):
        label_emb = self.label_emb(label)
        x_recon, mu, log_var = self.cvae(x, features, label_emb)

        return x_recon, mu, log_var

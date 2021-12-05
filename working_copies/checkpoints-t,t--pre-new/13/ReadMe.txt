Intiution same as 7
--beta-map 0.01 \
--beta-con 0.01 \
--neg-penalty 0.20 \
lr=1e-4
optimizer=Adam
epochs=40
total_loss+=vae_loss
loss from vae = 0.01*MSE + 0.1*KL
--save_dir checkpoints_13



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch import nn, optim
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self,D_in,H,H2,latent_dim):
        
        #Encoder
        super(Autoencoder,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2=nn.Linear(H,H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3=nn.Linear(H2,H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)
        
#         # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

#         # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)
        
#         # Decoder
        self.linear4=nn.Linear(H2,H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5=nn.Linear(H2,H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6=nn.Linear(H,D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))


        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        lambda1 = 0.01
        lambda2 = 0.1
        return lambda1*loss_MSE + lambda2*loss_KLD


class VAE:
    def __init__(self):
        print("VAE Initialised!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.D_in = 1024
        self.H = 512
        self.H2 = 256
        self.latent_dim = 128
        print(self.D_in)
        print(self.H)
        print(self.H2)
        print(self.latent_dim)

        self.model = Autoencoder(self.D_in, self.H, self.H2, self.latent_dim).to(device)
        # self.optimizer = optim.Adam (self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')


        self.loss_mse = customLoss()

        self.val_losses = []
        self.train_losses = []

        self.loss = 0
        self.min_loss = float('inf')
    
    
    def embeddings(self, inputs):
        # mean = torch.mean(inputs)
        # std = torch.std(inputs)
        # #normalize = torchvision.transforms.functional.normalize(mean,std)
        # # normalize = F.normalize(mean,std)
        # inputs = F.normalize(inputs,std,mean)
        dataloader = DataLoader(inputs, batch_size=256)
        # self.model.train()

        self.loss = 0
        mu_output = []
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_mse(recon_batch, data, mu, logvar)
            loss += loss.item()
            mu_tensor = mu   
            mu_output.append(mu_tensor)
            mu_result = torch.cat(mu_output, dim=0)

        # if(self.loss<self.min_loss):
        #     self.min_loss = loss
        #     self.save_checkpoint(prefix='min_loss')

        return loss, mu_result


    def save_checkpoint(self, prefix):
        path = f'{prefix}_vae-backprop.pth.tar'
        torch.save(
            { 
            'state_dict': self.model.state_dict(), 
            'lossMIN' : self.loss
            }, path)
        

    
    
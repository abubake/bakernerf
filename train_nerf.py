import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import get_rays
from model import Nerf
from ml_helpers import training

'''
Same functionality as train_nerf.ipynb, but all in one simple python file.
'''

# Parameters
batch_size = 1024
height = 400
width = 400
imgs = 100

o, d, target_px_values = get_rays('datasets/fox', mode='train')
dataloader = DataLoader(torch.cat((torch.from_numpy(o).reshape(-1, 3).type(torch.float),
                                   torch.from_numpy(d).reshape(-1, 3).type(torch.float),
                                   torch.from_numpy(target_px_values).reshape(-1, 3).type(torch.float)), dim=1),
                       batch_size=batch_size, shuffle=True)

dataloader_warmup = DataLoader(torch.cat((torch.from_numpy(o).reshape(imgs, height, width, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float),
                               torch.from_numpy(d).reshape(imgs, height, width, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float),
                               torch.from_numpy(target_px_values).reshape(imgs, height, width, 3)[:, 100:300, 100:300, :].reshape(-1, 3).type(torch.float)), dim=1),
                       batch_size=batch_size, shuffle=True)

test_o, test_d, test_target_px_values = get_rays('datasets/fox', mode='test')


wpth_file = 'nerf_models/cool_nerf.pth'
pth_file = 'nerf_models/cool_nerf.pth'

device = 'cuda'

tn = 8.
tf = 12.
nb_epochs = 1
lr = 1e-3
gamma = .5
nb_bins = 100

model = Nerf(hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=gamma)


training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, 1, dataloader_warmup, model_name=wpth_file, device=device)
plt.plot(training_loss)
plt.show()
training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, dataloader, model_name=pth_file, device=device)
plt.plot(training_loss)
plt.show()


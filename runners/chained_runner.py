import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image
import torch.nn as nn
from einops import rearrange

__all__ = ['ChainedRunner']


class ChainedRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)
        elif self.config.data.dataset == 'FashionMNIST':
            dataset = FashionMNIST(os.path.join(self.args.run, 'datasets', 'fashionmnist'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = FashionMNIST(os.path.join(self.args.run, 'datasets', 'fashionmnist_test'), train=False, download=True,
                                 transform=test_transform)

        elif self.config.data.dataset == 'CELEBA':
            if self.config.data.random_flip:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), download=True)
            else:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.ToTensor(),
                                 ]), download=True)

            test_dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                  ]), download=True)

        elif self.config.data.dataset == 'SVHN':
            dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                           transform=tran_transform)
            test_dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn_test'), split='test', download=True,
                                transform=test_transform)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        #score = CondRefineNetDilated(self.config).to(self.config.device)
        #score = torch.nn.DataParallel(score)
        score = LSTMModel(self.config).to(self.config.device)


        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)


        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1
                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.


                # flip X with probability p
                if not hasattr(self.config.data, 'flip_p'): 
                    self.config.data.flip_p = 0.0

                if self.config.data.flip_p > 0.0:
                    p = torch.rand(X.shape[0], device=self.config.device).view(-1,1,1,1).repeat(1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
                    flip_p = self.config.data.flip_p
                    X = (p > flip_p) * X + ~(p > flip_p) * (1-X)


                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                                                                    self.config.training.anneal_power)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images
        

    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        patch = self.config.model.patch
        seq_len = self.config.data.channels * (self.config.data.image_size ** 2) // patch
        batch_size = x_mod.size(0)

        scorenet.h_set, scorenet.c_set = [], []
        for j in range(scorenet.config.model.num_classes):
            sigma = scorenet.sigmas[j].view(-1,1,1).repeat(batch_size, 1, scorenet.patch)
            h0 = torch.zeros(scorenet.layer_dim, batch_size, scorenet.hidden_dim).to(scorenet.config.device).requires_grad_()
            c0 = torch.zeros(scorenet.layer_dim, batch_size, scorenet.hidden_dim).to(scorenet.config.device).requires_grad_()

            out, (h, c) = scorenet.lstm(sigma, (h0.detach(), c0.detach()))
            scorenet.h_set.append(h.clone().detach())
            scorenet.c_set.append(c.clone().detach())

        with torch.no_grad():
            for i in tqdm.tqdm(range(seq_len), total=seq_len, desc='annealed Langevin dynamics sampling'):
                x_mod = rearrange(x_mod, 'b c h w -> b (w h c)', c=self.config.data.channels, h=self.config.data.image_size, w=self.config.data.image_size)
                x_mod = rearrange(x_mod, 'b (s p) -> b s p', s=seq_len, p=patch)
                for label, sigma in enumerate(sigmas):
                    step_size = step_lr * (sigma / sigmas[-1]) ** 2

                    h = scorenet.h_set[label].clone().detach()
                    c = scorenet.c_set[label].clone().detach()
                    for s in range(n_steps_each):

                        out, (_h, _c) = scorenet.lstm(x_mod[:,i:(i+1),:], (h.clone().detach(), c.clone().detach()))
                        grad = scorenet.fc(out[:,0,:]).clone().detach()

                        noise = torch.randn_like(x_mod[:, i]) * np.sqrt(step_size * 2)
                        x_mod[:, i] = x_mod[:, i] + step_size * grad + noise

                # update h, c
                for label, sigma in enumerate(sigmas):
                    step_size = step_lr * (sigma / sigmas[-1]) ** 2

                    h = scorenet.h_set[label].clone().detach()
                    c = scorenet.c_set[label].clone().detach()
                    out, (_h, _c) = scorenet.lstm(x_mod[:,i:(i+1),:], (h.clone().detach(), c.clone().detach()))
                    scorenet.h_set[label] = _h
                    scorenet.c_set[label] = _c
                    
                x_mod = rearrange(x_mod, 'b s p -> b (s p)', s=seq_len, p=patch)
                x_mod = rearrange(x_mod, 'b (w h c) -> b c h w', c=self.config.data.channels, h=self.config.data.image_size, w=self.config.data.image_size)

                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
            return images


    def test(self, test_iter=30000, step_lr=0.00002):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        #score = CondRefineNetDilated(self.config).to(self.config.device)
        #score = torch.nn.DataParallel(score)
        score = LSTMModel(self.config).to(self.config.device)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()

        grid_size = 10
        n_steps_each = test_iter // int(self.config.model.num_classes * self.config.data.channels * self.config.data.image_size * self.config.data.image_size // self.config.model.patch)

        test_transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == 'MNIST':
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)
        elif self.config.data.dataset == 'FashionMNIST':
            test_dataset = FashionMNIST(os.path.join(self.args.run, 'datasets', 'fashionmnist_test'), train=False, download=True,
                                 transform=test_transform)

        elif self.config.data.dataset == 'CELEBA':
            test_dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                  ]), download=True)
            
        test_loader = DataLoader(test_dataset, batch_size=grid_size**2, shuffle=True,
                                 num_workers=4, drop_last=True)

        for i, (X, y) in enumerate(test_loader):
            image_from_dataset = X.to(self.config.device)
            image_grid = make_grid(X, nrow=grid_size)
            save_image(image_grid, os.path.join(self.args.image_folder, 'image_original.png'))
            image_grid = make_grid(1-X, nrow=grid_size)
            save_image(image_grid, os.path.join(self.args.image_folder, 'image_flip.png'))
            break


        # initialize the sample as random noise
        samples = torch.rand(grid_size ** 2, self.config.data.channels, self.config.data.image_size, self.config.data.image_size, device=self.config.device)
        all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, n_steps_each, step_lr)

        image_grid = make_grid(samples, nrow=grid_size)
        save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(0)))

        for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
            sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                    self.config.data.image_size)

            if self.config.data.logit_transform:
                sample = torch.sigmoid(sample)

            if i == len(all_samples) - 1:
                image_grid = make_grid(sample, nrow=grid_size)
                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))

        
        # initialize the sample as the original image
        all_samples = self.anneal_Langevin_dynamics(image_from_dataset.clone().detach(), score, sigmas, n_steps_each, step_lr)
        
        for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
            sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                    self.config.data.image_size)

            if self.config.data.logit_transform:
                sample = torch.sigmoid(sample)

            if i == len(all_samples) - 1:
                image_grid = make_grid(sample, nrow=grid_size)
                save_image(image_grid, os.path.join(self.args.image_folder, 'image_original_{}.png'.format(i)), nrow=grid_size)


        # initialize the sample as flipped image
        if self.config.data.flip_p > 0.0:
            flip_sample =  1 - image_from_dataset
            all_samples = self.anneal_Langevin_dynamics(flip_sample.clone().detach(), score, sigmas, n_steps_each, step_lr)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                        self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                if i == len(all_samples) - 1:
                    image_grid = make_grid(sample, nrow=grid_size)
                    save_image(image_grid, os.path.join(self.args.image_folder, 'image_flip_{}.png'.format(i)), nrow=grid_size)


class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.config = config

        self.hidden_dim = self.config.model.hidden_dim
        self.layer_dim = self.config.model.layer_dim
        self.patch = self.config.model.patch
        self.seq_len = self.config.data.channels * (self.config.data.image_size ** 2) // self.patch

        self.sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)
        
        self.lstm = nn.LSTM(self.patch, self.hidden_dim, self.layer_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.patch)


    def forward(self, x, labels):
        x = rearrange(x, 'b c h w -> b (w h c)', c=self.config.data.channels, h=self.config.data.image_size, w=self.config.data.image_size)
        x = rearrange(x, 'b (s p) -> b s p', s=self.seq_len, p=self.patch)

        sigma = self.sigmas[labels].view(-1,1,1).repeat(1, 1, self.patch)
        x = torch.cat((sigma,x), dim=1)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.config.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.config.device).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:,1:,:])
        
        out = rearrange(out, 'b s p -> b (s p)', s=self.seq_len, p=self.patch)
        out = rearrange(out, 'b (w h c) -> b c h w', c=self.config.data.channels, h=self.config.data.image_size, w=self.config.data.image_size)
        return out.contiguous()
    

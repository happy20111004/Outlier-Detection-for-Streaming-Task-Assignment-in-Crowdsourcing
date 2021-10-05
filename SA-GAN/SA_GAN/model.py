import time,os,sys


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .network import *
from .earlyStopping import EarlyStopping
from .eval_methods import *

##
class Discriminator_rec(nn.Module):

    def __init__(self, opt):
        super(Discriminator_rec, self).__init__()
        model = Encoder(opt.ngpu,opt,1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:])
        self.classifier = nn.Sequential(
            nn.Conv1d(opt.ndf*16, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

class Discriminator_latent(nn.Module):

    def __init__(self, opt):
        super(Discriminator_latent, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, opt.ndf, 4, 2, 1),
            nn.BatchNorm1d(opt.ndf),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1),
            nn.BatchNorm1d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1),
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv1d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1),
            nn.BatchNorm1d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv1d(opt.ndf * 8, opt.ndf * 16, 4, 2, 1),
            nn.BatchNorm1d(opt.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(opt.ndf * 16, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features


##
class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.encoder = Encoder(opt.ngpu,opt,opt.nz)
        self.decoder = Decoder(opt.ngpu,opt)

    def reparameter(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        latent_z = self.reparameter(mu, log_var)
        output = self.decoder(latent_z)
        return output, latent_z, mu, log_var

class SA(SA_MODEL):


    def __init__(self, opt, train_dataloader, val_dataloader, test_data, label, device, alpha_rec, alpha_lat):
        super(SA, self).__init__(opt)

        self.early_stopping = EarlyStopping(opt, patience=opt.patience, verbose=False)

        self.opt = opt
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_data = test_data
        self.label = label
        self.device = device

        self.alpha_rec = alpha_rec
        self.alpha_lat = alpha_lat
        self.wi_rec = 0.5
        self.wi_lat = 0.5

        self.train_batchsize = opt.train_batchsize
        self.val_batchsize = opt.val_batchsize
        self.test_batchsize = opt.test_batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator(opt).to(device)
        self.G.apply(weights_init)


        self.D_rec = Discriminator_rec(opt).to(device)
        self.D_rec.apply(weights_init)

        self.D_lat = Discriminator_latent(opt).to(device)
        self.D_lat.apply(weights_init)


        self.bce_criterion = nn.BCELoss()
        self.mse_criterion=nn.MSELoss()
        self.l1loss = nn.L1Loss()


        self.optimizer_D_rec = optim.Adam(self.D_rec.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        self.optimizer_D_lat = optim.Adam(self.D_lat.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))


        self.cur_epoch = 0
        self.input =  None
        self.output = None

        self.p_z = None
        self.real_label = 1.0
        self.fake_label= 0.0
        #
        #output of discriminator_rec
        self.out_d_rec_real = None
        self.feat_rec_real = None
        self.out_d_rec_fake = None
        self.feat_rec_fake = None

        #output of discriminator_lat
        self.out_d_lat_real = None
        self.feat_lat_real = None
        self.out_d_lat_fake = None
        self.feat_lat_fake = None

        #output of generator
        self.mu = None
        self.log_var = None
        self.out_g_fake = None
        self.latent_z = None

        #loss
        self.loss_d_rec = None
        self.loss_d_rec_real = None #(social)
        self.loss_d_rec_fake = None #(individual)

        self.loss_d_lat = None
        self.loss_d_lat_real = None #(social)
        self.loss_d_lat_fake = None #(individual)

        self.loss_g = None
        self.loss_g_rs = None
        self.loss_g_rec = None
        self.loss_g_lat = None

        if self.opt.update_method == 1:
            self.flag = 1
        elif self.opt.update_method == 2:
            self.flag = 2
        else:
            raise ValueError('update method is wrong')


    def train(self):

        self.train_hist = {}
        self.train_hist['per_epoch_time'] = []



        #print("Train model.")
        start_time = time.time()



        for epoch in range(self.niter):

            self.cur_epoch += 1

            self.train_epoch()

            val_error = self.validate()

            self.early_stopping(val_error, self.G, self.D_rec, self.D_lat)

            #print('epoch', epoch)

            if self.early_stopping.early_stop:
                #print('train finished with early stopping')
                break

        if not self.early_stopping.early_stop:
            #print('train finished with total epochs')
            self.save_weight_GD()

        total_train_time = time.time() - start_time

        self.save(self.train_hist)

        return total_train_time, np.mean(self.train_hist['per_epoch_time']), self.flag



    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        self.D_rec.train()
        self.D_lat.train()

        epoch_iter = 0

        for data_x, data_y in self.train_dataloader:

            self.train_batchsize = data_x.size(0)
            # self.total_steps += self.train_batchsize
            epoch_iter += 1

            self.input = data_x.permute([0,2,1]).float().to(self.device)
            self.output = data_y.float().to(self.device)
            self.p_z = torch.randn(self.input.size(0), 1, self.opt.nz).to(self.device)


            self.optimize()

            loss = self.get_errors()

            if (epoch_iter  % self.opt.print_freq) == 0:

                #print("Epoch: [%d] [%4d/%4d] D_rec_loss(R/F/ALL): %.6f/%.6f/%.6f, "
                #      "D_lat_loss(R/F/ALL): %.6f/%.6f/%.6f, "
                #      "G_loss(R/A/L/ALL): %.6f/%.6f/%.6f/%.6f" %
                #      ((self.cur_epoch), (epoch_iter), self.train_dataloader.dataset.__len__()
                #       // self.train_batchsize,
                #       loss["loss_d_rec_real"], loss["loss_d_rec_fake"], loss["loss_d_rec"],
                #       loss["loss_d_lat_real"], loss["loss_d_lat_fake"], loss["loss_d_lat"],
                #       loss["loss_g_rs"], loss["loss_g_rec"], loss["loss_g_lat"], loss["loss_g"]))
                pass


        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)



    ##

    def update_wi_1(self):

        if self.loss_d_rec_fake.item() < self.loss_d_rec_real.item():
            self.wi_rec = np.random.uniform(self.wi_rec, 1)

        elif self.loss_d_rec_fake.item() > self.loss_d_rec_real.item():
            self.wi_rec = np.random.uniform(0, self.wi_rec)

        else:
            self.wi_rec = self.wi_rec


        if self.loss_d_lat_fake.item() < self.loss_d_lat_real.item():
            self.wi_lat = np.random.uniform(self.wi_lat, 1)

        elif self.loss_d_lat_fake.item() > self.loss_d_lat_real.item():
            self.wi_lat = np.random.uniform(0, self.wi_lat)

        else:
            self.wi_lat = self.wi_lat



        if self.loss_g_rec.item() < self.loss_d_rec_real.item():
            self.wi_rec = np.random.uniform(self.wi_rec, 1)

        elif self.loss_g_rec.item() > self.loss_d_rec_real.item():
            self.wi_rec = np.random.uniform(0, self.wi_rec)

        else:
            self.wi_rec = self.wi_rec

        if self.loss_g_lat.item() < self.loss_d_lat_real.item():
            self.wi_lat = np.random.uniform(self.wi_lat, 1)

        elif self.loss_g_lat.item() > self.loss_d_lat_real.item():
            self.wi_lat = np.random.uniform(0, self.wi_lat)

        else:
            self.wi_lat = self.wi_lat


    def update_wi_2(self):

        self.wi_rec += self.alpha_rec * (self.loss_d_rec_fake.item() - self.loss_d_rec_real.item()) / \
                       (self.loss_d_rec_fake.item() + self.loss_d_rec_real.item())

        self.wi_lat += self.alpha_lat * (self.loss_d_lat_fake.item() - self.loss_d_lat_real.item()) / \
                       (self.loss_d_lat_fake.item() + self.loss_d_lat_real.item())


        self.wi_rec += self.alpha_rec * (self.loss_g_rec.item() - self.loss_d_rec_real.item()) / \
                       (self.loss_g_rec.item() + self.loss_d_rec_real.item())

        self.wi_lat += self.alpha_lat * (self.loss_g_lat.item() - self.loss_d_lat_real.item()) / \
                       (self.loss_g_lat.item() + self.loss_d_lat_real.item())


    def optimize(self):

        self.update_d_rec()

        self.update_d_lat()

        self.update_g()

        if self.flag == 1:
            self.update_wi_1()
        elif self.flag == 2:
            self.update_wi_2()
        else:
            raise ValueError('no such updating method')

        # If D loss too low, then re-initialize netD
        if self.loss_d_rec.item() < 5e-6:
            self.reinitialize_netd_rec()
        if self.loss_d_lat.item() < 5e-6:
            self.reinitialize_netd_lat()

    def update_d_rec(self):
        ##

        self.D_rec.zero_grad()
        # --
        # real
        self.out_d_rec_real, self.feat_rec_real = self.D_rec(self.input)

        # --
        # fake
        self.out_g_fake, self.latent_z, _, _ = self.G(self.input)
        self.out_d_rec_fake, self.feat_rec_fake = self.D_rec(self.out_g_fake.detach())
        # --

        self.loss_d_rec_real = self.bce_criterion(self.out_d_rec_real,
                                    torch.full((self.train_batchsize,), self.real_label, device=self.device))
        self.loss_d_rec_fake = self.bce_criterion(self.out_d_rec_fake,
                                    torch.full((self.train_batchsize,), self.fake_label, device=self.device))


        self.loss_d_rec = self.wi_rec * self.loss_d_rec_real + (1-self.wi_rec) * self.loss_d_rec_fake
        self.loss_d_rec.backward()
        # nn.utils.clip_grad_norm_(self.D_rec.parameters(), max_norm=10)
        self.optimizer_D_rec.step()

    def update_d_lat(self):
        ##

        self.D_lat.zero_grad()
        # --
        # real
        self.out_d_lat_real, self.feat_lat_real = self.D_lat(self.p_z)

        # --
        # fake
        self.out_g_fake, self.latent_z, _, _ = self.G(self.input)
        self.latent_z = self.latent_z.permute([0, 2, 1])

        self.out_d_lat_fake, self.feat_lat_fake = self.D_lat(self.latent_z.detach())
        # --

        self.loss_d_lat_real = self.bce_criterion(self.out_d_lat_real,
                                    torch.full((self.train_batchsize,), self.real_label, device=self.device))
        self.loss_d_lat_fake = self.bce_criterion(self.out_d_lat_fake,
                                    torch.full((self.train_batchsize,), self.fake_label, device=self.device))


        self.loss_d_lat = self.wi_lat * self.loss_d_lat_real + (1-self.wi_lat) * self.loss_d_lat_fake
        self.loss_d_lat.backward()
        # nn.utils.clip_grad_norm_(self.D_rec.parameters(), max_norm=10)
        self.optimizer_D_lat.step()

    def update_g(self):

        self.G.zero_grad()

        self.out_g_fake, self.latent_z, self.mu, self.log_var = self.G(self.input)

        #dis_rec
        _, self.feat_rec_fake = self.D_rec(self.out_g_fake)
        _, self.feat_rec_real = self.D_rec(self.input)

        #dis_lat
        self.latent_z = self.latent_z.permute([0,2,1])
        _, self.feat_lat_fake = self.D_lat(self.latent_z)
        _, self.feat_lat_real = self.D_lat(self.p_z)

        self.loss_g_rs = self.l1loss(self.out_g_fake.permute([0,2,1]), self.output)
        self.loss_g_rec = self.mse_criterion(self.feat_rec_fake, self.feat_rec_real)  # loss for feature matching
        self.loss_g_lat = self.mse_criterion(self.feat_lat_fake, self.feat_lat_real)  # constrain x' to look like x


        self.loss_g = self.loss_g_rs + self.opt.w_rec  * self.loss_g_rec + self.opt.w_lat * self.loss_g_lat
        self.loss_g.backward()
        self.optimizer_G.step()

    def reinitialize_netd_rec(self):
        """ Initialize the weights of netD
        """
        self.D_rec.apply(weights_init)
        # print('Reloading d_rec net')

    def reinitialize_netd_lat(self):
        """ Initialize the weights of netD
        """
        self.D_lat.apply(weights_init)
        # print('Reloading d_lat net')

    ##
    def get_errors(self):

        loss = {'loss_d_rec' : self.loss_d_rec.item(),
                'loss_d_rec_fake': self.loss_d_rec_fake.item(),
                'loss_d_rec_real': self.loss_d_rec_real.item(),

                'loss_d_lat' : self.loss_d_rec.item(),
                'loss_d_lat_fake': self.loss_d_lat_fake.item(),
                'loss_d_lat_real': self.loss_d_lat_real.item(),

                'loss_g': self.loss_g.item(),
                'loss_g_lat': self.loss_g_lat.item(),
                'loss_g_rs': self.loss_g_rs.item(),
                'loss_g_rec': self.loss_g_rec.item(),
                  }

        return loss

        ##

    def validate(self):
        '''
        validate by validation loss
        :return: auc
        '''
        l1loss = nn.L1Loss()
        self.G.eval()
        self.D_lat.eval()
        self.D_rec.eval()

        loss = []
        with torch.no_grad():
            for data_x, data_y in self.val_dataloader:
                input_data = data_x.permute([0,2,1]).float().to(self.device)
                output = data_y.float().to(self.device)
                fake, _, _, _ = self.G(input_data)
                loss.append(l1loss(output, fake.permute([0,2,1])).cpu().numpy())

            val_loss = np.mean(loss)

        return val_loss

    def predict(self, dataloader, test_label, scale=True):

        with torch.no_grad():

            collector = []
            pred_time = []

            for i, data in enumerate(dataloader, 0):

                start_time = time.time()

                input_data = data.permute([0,2,1]).float().to(self.device)

                fake, _, _, _ = self.G(input_data)

                fake = fake.type(torch.DoubleTensor)
                data = data.type(torch.DoubleTensor)

                rec_error = torch.sum(torch.abs((fake.permute([0,2,1]) - data)), dim=2)

                collector.append(rec_error[:, -1])#取每个窗口最后一个点的重构误差

                pred_time.append(time.time() - start_time)

            score = np.concatenate(collector, axis=0)

            # Scale error vector between [0, 1]
            if scale:
                score = (score - np.min(score)) / (np.max(score) - np.min(score))

            y_ = test_label

            y_pred = score

            if y_ is not None and len(y_) > len(y_pred):
                y_ = y_[-len(y_pred):]

            return y_, y_pred, np.mean(pred_time)

    def eval_result(self, dataloader, test_label, save_dir, dataset, file):

        self.G.eval()

        y_t, y_pred, test_time = self.predict(dataloader, test_label, scale=True)

        label_dir = Path(save_dir) / "label" / dataset
        pred_dir = Path(save_dir) / "pred" / dataset
        stem = Path(file).stem
        if not label_dir.exists():
            label_dir.mkdir(parents=True)
        if not pred_dir.exists():
            pred_dir.mkdir(parents=True)
        np.save(label_dir / stem, y_t)
        np.save(pred_dir / stem, y_pred)

        t, th = bf_search(y_pred, y_t)

        #print('best_f1:', t[0], 'pre:', t[1], 'rec:', t[2], 'TP:', t[3], 'TN:', t[4], 'FP:', t[5], 'FN:', t[6],
        #      'threshold:', th)

        return t[0], t[1], t[2], t[3], t[4], t[5], t[6], test_time

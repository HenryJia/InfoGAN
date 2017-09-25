import os

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm_notebook

from layers import Linear

import pylab as plt

class InfoGAN:
    def __init__(self, gen, dis, embedding_len, z_len, c1_len, c2_len, c3_len):
        assert c1_len is not None and c1_len > 1, 'Must have a categorical code'

        self.gen = gen.cuda()
        self.dis = dis.cuda()

        self.embedding_len = embedding_len
        self.z_len = z_len
        self.c1_len = c1_len
        self.c2_len = c2_len
        self.c3_len = c3_len

        self.Q_cat = Linear(embedding_len, c1_len).cuda()
        self.qcat_optim = optim.Adam(self.Q_cat.parameters(), lr = 2e-4)
        if c2_len:
            self.Q_con = Linear(embedding_len, c2_len).cuda()
            self.qcon_optim = optim.Adam(self.Q_con.parameters(), lr = 2e-4)
        if c3_len:
            self.Q_bin = Linear(embedding_len, c3_len).cuda()
            self.qbin_optim = optim.Adam(self.Q_bin.parameters(), lr = 2e-4)

        self.g_optim = optim.Adam(self.gen.parameters(), lr = 1e-3)
        self.d_optim = optim.Adam(self.dis.parameters(), lr = 2e-4)


    def train_all(self, train_loader):
        nll = nn.NLLLoss().cuda()
        mse = nn.MSELoss().cuda()
        bce = nn.BCELoss().cuda()

        plt.figure(0, figsize = (32, 32))
        for epoch in range(100):
            pb = tqdm_notebook(total = train_loader.dataset.data_tensor.size()[0])
            pb.set_description('Epoch ' + str(epoch + 1))
            for i, (data, targets) in enumerate(train_loader):
                ones = Variable(torch.ones(data.size()[0], 1)).cuda()
                zeros = Variable(torch.zeros(data.size()[0], 1)).cuda()

                z_dict = self.get_z(data.size()[0])
                z = torch.cat([z_dict[k] for k in z_dict.keys()], dim = 1)

                data = Variable(data.float().cuda(async = True)) / 255
                targets = Variable(targets.float().cuda(async = True))

                '''
                So here's how it's gonna work. This is the most efficient solution I could come up with as of now.
                We will run a forward and backward pass on the real data. This will produce gradients for Q_cat and dis.
                Then we will run a forward on the generated data and do a backward pass for the latent code objective.
                This will produce gradients of the latent code objective for everything.
                We will then run the update for the discriminator and the latent code output layers.
                Lastly, we will run the backward pass for the adversarial objective for the generator, since we don't want its
                gradients interfering with the discriminator. We will then do the generator's update.
                '''

                # Forward pass on real MNIST
                out_dis, hid = self.dis(data)
                c1 = F.log_softmax(self.Q_cat(hid))
                loss_dis = mse(out_dis, ones) - torch.sum(targets * c1) / (torch.sum(targets) + 1e-3) # Loss for real MNIST

                # Forward pass on generated MNIST
                out_gen = self.gen(z)
                out_dis, hid = self.dis(out_gen)

                # Loss for generated MNIST
                loss_dis = loss_dis + mse(out_dis, zeros)
                loss_dis = loss_dis 

                # Zero gradient buffers for gen and Q_cat and backward pass
                self.dis.zero_grad()
                self.Q_cat.zero_grad()
                loss_dis.backward(retain_graph = True) # We need PyTorch to retain the graph buffers so we can run backward again later
                self.d_optim.step() # Apply the discriminator's update now since we have to delete its gradients later

                # And backward pass and loss for generator and update
                self.gen.zero_grad()
                loss_gen = mse(out_dis, ones)
                loss_gen.backward(retain_graph = True)
                self.dis.zero_grad() # Don't want the gradients of the generator's objective in the discriminator

                # Forward pass and loss for latent codes
                loss_q = 0

                c1 = F.log_softmax(self.Q_cat(hid))
                loss_q += nll(c1, torch.max(z_dict['cat'], dim = 1)[1])

                if self.c2_len:
                    c2 = self.Q_con(hid)
                    loss_q += 0.5 * mse(c2, z_dict['con']) # Multiply by 0.5 as we treat targets as Gaussian (and there's a coefficient of 0.5 when we take logs)
                    self.Q_con.zero_grad() # Zero gradient buffers before the backward pass
                if self.c3_len:
                    c3 = F.sigmoid(self.Q_bin(hid))
                    loss_q += bce(c3, z_dict['bin'])
                    self.Q_bin.zero_grad() # Zero gradient buffers before the backward pass

                # Backward pass for latent code objective
                loss_q.backward()

                # Do the updates for everything
                self.d_optim.step()
                self.g_optim.step()
                self.qcat_optim.step()

                if self.c2_len:
                    self.qcon_optim.step()
                if self.c3_len:
                    self.qbin_optim.step()



                pb.update(data.size()[0])
                pb.set_postfix(loss_dis = loss_dis.cpu().data.numpy()[0], loss_gen = loss_gen.cpu().data.numpy()[0], loss_q = loss_q.cpu().data.numpy()[0])

            pb.close()
            plt.subplot(10, 10, epoch + 1)
            plt.imshow(out_gen.cpu().data.numpy()[0, 0], cmap = 'gray')


    # Generate a noise vector and random latent codes for generator
    def get_z(self, length, sequential = False):
        weights = torch.Tensor([0.1] * 10)

        z = {}
        if self.z_len:
            z['z'] = Variable(torch.randn(length, self.z_len)).cuda()

        if self.c1_len:
            if sequential:
                cat_noise = Variable(torch.arange(0, self.c1_len).repeat(length // self.c1_len).long()).cuda()
            else:
                cat_noise = Variable(torch.multinomial(weights, num_samples = length, replacement = True)).cuda().view(-1)
            onehot_noise = Variable(torch.zeros(length, self.c1_len)).cuda()
            onehot_noise.data.scatter_(1, cat_noise.data.view(-1, 1), 1)
            z['cat'] = onehot_noise

        if self.c2_len:
            #z['con'] = Variable(torch.randn(length, c2_len)).cuda()
            z['con'] = Variable(torch.rand(length, self.c2_len)).cuda() * 2 - 1

        if self.c3_len:
            z['bin'] = Variable(torch.bernoulli(0.5 * torch.ones(length, self.c3_len))).cuda().float()

        return z


    def run_dis(self, x):
        out = []
        out_dis, hid = self.dis(x)
        out += [out_dis]
        if self.c1_len:
            out += [F.softmax(self.Q_cat(hid))]
        if self.c2_len:
            out += [self.Q_con(hid)]
        if self.c3_len:
            out += [F.sigmoid(Q_bin(hid))]

        return out


    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.gen.state_dict(), directory + 'gen.torch')
        torch.save(self.dis.state_dict(), directory + 'dis.torch')
        if self.c1_len:
            torch.save(self.Q_cat.state_dict(), directory + 'qcat.torch')
        if self.c2_len:
            torch.save(self.Q_con.state_dict(), directory + 'qcon.torch')
        if self.c3_len:
            torch.save(self.Q_bin.state_dict(), directory + 'qbin.torch')


    def load(self, directory):
        self.gen.load_state_dict(torch.load(directory + 'gen.torch'))
        self.dis.load_state_dict(torch.load(directory + 'dis.torch'))
        if self.c1_len:
            self.Q_cat.load_state_dict(torch.load(directory + 'qcat.torch'))
        if self.c2_len:
            self.Q_con.load_state_dict(torch.load(directory + 'qcon.torch'))
        if self.c3_len:
            self.Q_bin.load_state_dict(torch.load(directory + 'qbin.torch'))

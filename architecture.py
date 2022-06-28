
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
sigmoid = nn.Sigmoid()


class gan(nn.Module):
    """
           CNN Model for GAN
    """
    def __init__(self, device, learning_rate, batch_size):
        super(gan, self).__init__()
        self.learning_rate = learning_rate
        self.device = device
        self.batch_size = batch_size
        self.real = torch.ones((batch_size)).to(device)
        self.fake = torch.zeros((batch_size)).to(device)
        self.lossFN = nn.BCELoss()
        self.make_belive_real = torch.ones((batch_size)).to(device)

        self.generator = nn.Sequential(
            nn.Linear(16, 64),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (-1, 1, 1)),
            nn.ConvTranspose2d(64, 32, 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2, padding=1),  # 16
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, stride=1, padding=1),  # 16
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # 16
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 4
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, 4),  # 1 x 64 x batch size
            nn.Flatten(),
            nn.Sigmoid()
        )

    def sampling(self, dim):
        sample = torch.randn((dim, 16)).to(self.device)
        return sample

    def d_forward(self, x):
        """
            z: random noise passed to generator
            g_z: generated image (fake)
            d_g_z: pass fake image to discriminator
            d_x: pass real image to discriminator

        """
        z = self.sampling(x.shape[0])
        g_z = self.generator(z)
        d_g_z = self.discriminator(g_z)
        d_g_z = (d_g_z).squeeze()
        d_x = self.discriminator(x)
        d_x = (d_x).squeeze()

        return d_g_z, d_x, g_z

    def g_forward(self, dim):
        '''
            z: random noise passed to generator
            g_z: generated image (fake)
            d_g_z: pass fake image to discriminator
        '''
        z = self.sampling(dim)
        g_z = self.generator(z)
        d_g_z = self.discriminator(g_z)
        # print("d_g_z", d_g_z)
        d_g_z = (d_g_z).squeeze()
        # print("d_x", d_x)

        return d_g_z, g_z

    def discriminator_loss(self, d_x, d_g_z):
        '''
            returns real and fake loss
            params:
            d_x: output of discriminator when real image is passed through it
            d_g_z: output of discriminator when generated image is passed through it

        '''
        dim = d_x.shape[0]
        real_loss = self.lossFN(d_x, self.real[0:dim]).mean()
        fake_loss = self.lossFN(d_g_z, self.fake[0:dim]).mean()
        # loss =(torch.log(d_x) + torch.log(1-d_g_z)).mean()
        # print("dloss",loss)
        return real_loss, fake_loss

    def generator_loss(self, d_g_z):
        '''
            returns generator loss to make believe that fake images are real to fool discriminator
            params:
            d_g_z: output of discriminator when generated image is passed through it
        '''
        loss = self.lossFN(d_g_z, self.make_belive_real[0:d_g_z.shape[0]]).mean()
        return loss

import torch
import torch.nn as nn

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 96

# Size of feature maps in generator
ngf = 32

# Size of feature maps in discriminator
ndf = 64

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
fixed_noise = torch.randn(64, nz, 1, 1, device=device)


class gan(nn.Module):
    '''
    CNN model for GAN
    '''
    def __init__(self, device, learning_rate, batch_size):
        super(gan, self).__init__()
        self.learning_rate = learning_rate
        self.device = device
        self.batch_size = batch_size
        self.lossFN = nn.BCELoss()

        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.discriminator = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def sampling(self, dim):
        noise = torch.randn(dim, nz, 1, 1, device=device)
        return noise

    def d_forward(self, input):
        return self.discriminator(input).squeeze()

    def g_forward(self, dim, f_b):
        '''
            z: random noise passed to generator
            g_z: generated image (fake)
        '''
        if f_b == True:
            z = fixed_noise
        else:
            z = self.sampling(dim)
        g_z = self.generator(z)
        return g_z
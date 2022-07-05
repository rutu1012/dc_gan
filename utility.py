import torch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def view_images(img_list, real_batch, epoch):
    '''
    This function displays the result images
    param
    img_list: list containing per epoch results(generated img) obtained after training
    real_batch: list containing per epoch results(real img) obtained after training
    '''
    print(f"Epoch: {epoch}")
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0))) # Plot the fake images from the last epoch
    plt.show()


def cost_graph(g_loss, d_loss,title):
    '''
    This function plots the loss graph of discriminator and generator
    d_loss: discriminator loss per epoch
    g_loss: generator loss per epoch
    '''
    plt.figure(figsize=(18, 6))
    plt.suptitle(title)
    ax = plt.subplot(1, 2, 1)
    plt.plot(d_loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    ax.set_title("Discriminator Loss")

    ax = plt.subplot(1, 2, 2)
    plt.plot(g_loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    ax.set_title("Generator Loss")
    plt.show()

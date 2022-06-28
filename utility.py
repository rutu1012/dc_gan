from matplotlib import pyplot as plt
import numpy as np


def cost_graph(d_loss, g_loss, title):
    '''
    This function plots the loss graph of discriminator and generator
    d_loss: discriminator loss per epoch
    g_loss: generator loss per epoch
    title: title for the graph
    '''
    plt.plot(d_loss, color='blue', label='Discriminator loss')
    plt.plot(g_loss, color='red', label='Generator loss')
    plt.ylabel('loss')
    plt.xlabel('epochs ')
    plt.title(title)
    plt.legend()
    plt.show()


def view_images(output, epoch):
    '''
        This function displays the result images
        param output: list containing per epoch results(real and generated) obtained after training/testing
    '''

    plt.figure(figsize=(120, 20))
    plt.suptitle("Epoch: %i" % (epoch + 1))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.title("Class " + str(i + 1))
        gen = output[1][i].cpu().detach()
        # print(gen.numpy().shape)
        plt.imshow(np.rot90(gen.numpy().reshape((3, 32, 32)).T, 3))
    plt.show()

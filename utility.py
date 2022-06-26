from matplotlib import pyplot as plt
import numpy as np


# def cost_graph(loss_list, title):
#     plt.plot(loss_list)
#     plt.ylabel('loss')
#     plt.xlabel('epochs')
#     plt.title(title)
#     plt.show()

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

# def view_images(output, num_epochs):
#
#     for k in range(0, num_epochs, num_epochs//3):
#
#         ori_imgs = output[k][1]
#         recon = output[k][2]
#         num = 10
#
#         plt.figure(figsize=(18, 5))
#         plt.suptitle("Epoch: %i" % (k + 1))
#
#         for i in range(num):
#             # plt.subplot(2, n, i+1)
#             ax = plt.subplot(2, num, i + 1 + num)
#             plt.imshow(recon[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
#             plt.xticks([])
#             plt.yticks([])  # removing axes
#
#
#         plt.show()
def view_images(output, epoch):
    # print(f"Epoch: {epoch}")
    # plt.figure(figsize=(120, 20))
    # for i in range(10):
    #     plt.subplot(1, 10, i + 1)
    #     plt.title("Class " + str(i + 1))
    #     plt.imshow(np.rot90(output[0][i].cpu().detach().numpy().reshape((3, 32, 32)).T, 3))
    # plt.show()
    plt.figure(figsize=(120, 20))
    plt.suptitle("Epoch: %i" % (epoch+1))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.title("Class " + str(i + 1))
        gen = output[1][i].cpu().detach()
        # print(gen.numpy().shape)
        plt.imshow(np.rot90(gen.numpy().reshape((3, 32, 32)).T, 3))
    plt.show()

import torch
import torch
from architecture import gan
from train import training
from utility import cost_graph, view_images, weights_init
import loader
import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # to downld cifar10

batch_size = 64
num_epochs = 2
train_loader = loader.trainLoader(batch_size)
test_loader = loader.testLoader(batch_size)
lr = 2 * 1e-4


def main(training_data, testing_data):
    '''
        This is the main function
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = gan(device, lr, batch_size).to(device)
    model.apply(weights_init)
    g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=lr)

    FILE = "dcgan.pt"
    gen_img_train, real_batch_f, d_loss, g_loss = training(model, train_loader, num_epochs, g_optimizer, d_optimizer)
    cost_graph(g_loss, d_loss, "DCGAN Train Loss")
    s = input("Model has been trained do you want to save it? (y/n): ").lower()
    if s == 'y':
        torch.save(model.state_dict(), FILE)  # saves the trained model at the specified path


main(train_loader, test_loader)

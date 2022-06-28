import torch
import torch
from architecture import gan
from trainNtest import training, testing
from utility import cost_graph, view_images
import loader
import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # to downld cifar10

batch_size = 64
num_epochs = 10
train_loader = loader.trainLoader(batch_size)
test_loader = loader.testLoader(batch_size)
lr = 1e-5


def main(training_data, testing_data, num_epochs):
    '''
        This is the main function
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = gan(device, lr, batch_size).to(device)
    g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=lr)

    FILE = "dcgan.pt"
    ch = input("Press l to load model, t to train model: ").lower()  # asks user if they want to train the model or load the already saved model
    if ch == 'l':
        model.load_state_dict(torch.load(FILE, map_location=device))  # loads the model
        # test the loaded model on the Test data
        test_outputs, dloss, gloss = testing(model, test_loader, num_epochs)
        cost_graph(dloss, gloss, "DCGAN  Test Loss")

    elif ch == 't':
        train_outputs, d_loss, g_loss = training(model, train_loader, num_epochs, g_optimizer, d_optimizer)
        cost_graph(d_loss, g_loss, "DCGAN  Train Loss")
        s = input("Model has been trained do you want to save it? (y/n): ").lower()
        if s == 'y':
            torch.save(model.state_dict(), FILE)  # saves the trained model at the specified path
        test_outputs, dloss, gloss = testing(model, test_loader, num_epochs)
        cost_graph(dloss, gloss, "DCGAN  Test Loss")


main(train_loader, test_loader, num_epochs)

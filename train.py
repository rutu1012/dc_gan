
from utility import view_images
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.utils as vutils

real_label = 1
fake_label = 0
lossFN = nn.BCELoss()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def genDisc_train(model, img, g_opt, d_opt):
    '''
    This function trains the Discriminator and generator.
    returns
    :discriminator loss
    :generator loss

    parameters:
    model: CNN Gan model
    img: input image
    d_opt: discriminator optimizer
    g_opt: generator optimizer
    '''
    img = img.to(model.device)
    dim = img.size(0)
    label = torch.full((dim,), real_label, dtype=torch.float, device=device)

    # discriminator training with real data
    d_opt.zero_grad()
    d_x = model.d_forward(img).view(-1)
    realD_loss = lossFN(d_x, label)
    realD_loss.backward()

    # generator training given noise
    g_z = model.g_forward(dim, f_b=False)

    # discriminator on fake batch input

    d_g_z = model.d_forward(g_z.detach())
    label.fill_(fake_label)  # fills label with 0

    fakeD_loss = lossFN(d_g_z, label)
    fakeD_loss.backward()

    Dtotal_loss = realD_loss + fakeD_loss
    d_opt.step()

    # Update G network: maximize log(D(G(z)))
    g_opt.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # perform another forward pass of all-fake batch through D
    d_g_z = model.d_forward(g_z)
    Gloss = lossFN(d_g_z, label)
    Gloss.backward()
    g_opt.step()

    return Gloss, Dtotal_loss


def training(model, train_loader, num_epochs, g_opt, d_opt):
    '''
        This function trains the GAN model.
        Parameters-
        model: CNN GAN model
        train_loader: DataLoader object that iterates through the train set
        g_opt: optimizer object to be used for training the generator
        d_opt: optimizer object to be used for training the discriminator
        num_epochs: number of epochs
        Returns:
        discriminator and generator loss
        img_list_f: list containing generated images of last epoch.
        real_batch_f: list containing real images of last epoch.
        '''
    dloss = []
    gloss = []
    img_list = []
    real_batch = []
    img_list_f = []
    real_batch_f = []
    for epoch in range(num_epochs):
        running_discriminator_loss = 0.0
        running_generator_loss = 0.0
        # #discriminator

        for i, (img, _) in enumerate(tqdm(train_loader)):
            # print("training discriminator")

            GLoss, DLoss = genDisc_train(model, img, g_opt, d_opt)
            running_generator_loss += GLoss.item()
            running_discriminator_loss += DLoss.item()

            if i % 50 == 0:
                with torch.no_grad():
                    fake = model.g_forward(img.size(0), f_b=True)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                real_batch.append(img)
            if epoch == num_epochs - 1:
                with torch.no_grad():
                    fake = model.g_forward(img.size(0), f_b=True)
                img_list_f.append(fake)
                real_batch_f.append(img)

        print(f"Epoch {epoch + 1}\tDiscrimnative Loss: {running_discriminator_loss / len(train_loader)} \tGenerative loss: {running_generator_loss / len(train_loader)}")

        view_images(img_list, real_batch, epoch)
        dloss.append(running_discriminator_loss / len(train_loader))
        gloss.append(running_generator_loss / len(train_loader))

    return img_list_f, real_batch_f, dloss, gloss



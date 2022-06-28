import torch
from tqdm import tqdm
from utility import view_images


def d_train(model, img, d_opt):
    """
        This function trains the Discriminator.
        returns discriminator loss
        parameters:
        model: CNN Gan model
        img: input image
        d_opt: discriminator optimizer

        """
    img = img.to(model.device)
    d_opt.zero_grad()
    d_g_z, d_x, g_z = model.d_forward(img)
    real_loss, fake_loss = model.discriminator_loss(d_x, d_g_z)
    real_loss.backward()
    fake_loss.backward()
    loss = real_loss + fake_loss
    d_opt.step()
    return loss


def g_train(model, img, g_opt):
    '''
        This function trains the Generator.
        returns:
        :generator loss
        :real and fake image
        parameters:
        model: CNN Gan model
        img: input image
        d_opt: generator optimizer
    '''
    img = img.to(model.device)
    g_opt.zero_grad()
    d_g_z, g_z = model.g_forward(img.shape[0])
    loss = model.generator_loss(d_g_z)
    loss.backward()
    g_opt.step()
    # train_op.append((img, g_z))
    return loss, g_z, (img, g_z)


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
        train_outputs: list containing generated and real images.
    '''
    dloss = []
    gloss = []
    train_outputs = []
    for epoch in range(num_epochs):
        running_discriminator_loss = 0.0
        running_generator_loss = 0.0
        # #discriminator
        for i, (img, _) in enumerate(tqdm(train_loader)):
            # print("training discriminator")
            running_discriminator_loss += d_train(model, img, d_opt).item()
            # #generator
            g_loss, g_z, op_li = g_train(model, img, g_opt)
            running_generator_loss += g_loss.item()
        print(f"Epoch {epoch + 1}\tDiscrimnative Loss: {running_discriminator_loss / len(train_loader)} \tGenerative loss: {running_generator_loss / len(train_loader)}")
        # view_images(train_outputs[epoch],10)
        if epoch % 3 == 0:
            view_images(op_li, epoch)
        train_outputs.append(op_li)
        dloss.append(running_discriminator_loss / len(train_loader))
        gloss.append(running_generator_loss / len(train_loader))
    return train_outputs, dloss, gloss


def d_test(model, test_loader):
    '''
        This function tests the Discriminator.
        returns discriminator loss
        parameters:
        model: CNN Gan model
        test_loader: DataLoader object that iterates through the test set
    '''
    with torch.no_grad():
        for i, (img, _) in enumerate(tqdm(test_loader)):
            img = img.to(model.device)
            d_g_z, d_x, g_z = model.d_forward(img)
            real_loss, fake_loss = model.discriminator_loss(d_x, d_g_z)
    loss = real_loss + fake_loss
    return loss


def g_test(model, test_loader):
    '''
      This function tests the Generator.
      returns
      :generator loss
      :real and fake image
      parameters:
      model: CNN Gan model
      test_loader: DataLoader object that iterates through the test set
    '''
    for i, (img, _) in enumerate(tqdm(test_loader)):
        img = img.to(model.device)
        d_g_z, g_z = model.g_forward(img.shape[0])
        loss = model.generator_loss(d_g_z)

    return loss, (img, g_z)


def testing(model, test_loader, num_epochs):
    '''
        This function tests the GAN model.
        Parameters-
        model: CNN GAN model
        test_loader: DataLoader object that iterates through the test set
        num_epochs: number of epochs
        Returns:
        discriminator and generator loss
        test_outputs: list containing generated and real images.
    '''
    test_outputs = []
    dloss = []
    gloss = []
    with torch.no_grad():
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_discriminator_loss = 0.0
            running_generator_loss = 0.0
            # #discriminator
            print("testing discriminator")
            running_discriminator_loss += d_test(model, test_loader).item()
            # #generator
            print("testing generator")
            g_loss, op_li = g_test(model, test_loader)
            running_generator_loss += g_loss.item()
            print(f"Epoch {epoch + 1}\tDiscrimnative Loss: {running_discriminator_loss / len(test_loader)} \tGenerative loss: {running_generator_loss / len(test_loader)}")
            test_outputs.append(op_li)
            if epoch % 3 == 0:
                view_images(op_li, epoch)
        dloss.append(running_discriminator_loss / len(test_loader))
        gloss.append(running_generator_loss / len(test_loader))

    return test_outputs, dloss, gloss

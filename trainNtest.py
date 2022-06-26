import torch
from tqdm import tqdm
from utility import view_images

def d_train(model, img, d_opt):
    img = img.to(model.device)
    d_opt.zero_grad()
    d_g_z, d_x, g_z = model.d_forward(img)
    real_loss, fake_loss = model.discriminator_loss(d_x, d_g_z)
    real_loss.backward()
    fake_loss.backward()
    loss = real_loss + fake_loss
    # for group in d_opt.param_groups:
    #     for p in group['params']:
    #         if p.grad is not None:
    #             p.grad = -1*p.grad
    d_opt.step()
    return loss


def g_train(model, img, g_opt):
    img = img.to(model.device)
    g_opt.zero_grad()
    d_g_z, g_z = model.g_forward(img.shape[0])
    loss = model.generator_loss(d_g_z)
    loss.backward()
    g_opt.step()
    # train_op.append((img, g_z))
    return loss, g_z, (img,g_z)


def training(model, train_loader, num_epochs, g_opt, d_opt):
    k = 5
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
            # print("Generator training started")
            # print("training generator")
            g_loss, g_z, op_li = g_train(model, img, g_opt)
            running_generator_loss += g_loss.item()
        print(f"Epoch {epoch+1}\tDiscrimnative Loss: {running_discriminator_loss / len(train_loader)} \tGenerative loss: {running_generator_loss / len(train_loader)}")
        # view_images(train_outputs[epoch],10)
        train_outputs.append(op_li)
        view_images(op_li, epoch)
        dloss.append(running_discriminator_loss / len(train_loader))
        gloss.append(running_generator_loss / len(train_loader))
    return train_outputs, dloss, gloss


def d_test(model, test_loader):
    with torch.no_grad():
        for i, (img, _) in enumerate(tqdm(test_loader)):
            img = img.to(model.device)
            d_g_z, d_x, g_z = model.d_forward(img)
            real_loss, fake_loss = model.discriminator_loss(d_x, d_g_z)
    loss = real_loss + fake_loss
    return loss


def g_test(model, test_loader):
    for i, (img, _) in enumerate(tqdm(test_loader)):
        img = img.to(model.device)
        d_g_z, g_z = model.g_forward(img.shape[0])
        loss = model.generator_loss(d_g_z)
    # est_outputs.append((img, g_z))t
    return loss, (img,g_z)


def testing(model, test_loader, num_epochs):
    k = 5
    total_loss = []
    test_outputs=[]
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
            # print("Generator training started")
            print("testing generator")
            g_loss, op_li = g_test(model, test_loader)
            running_generator_loss += g_loss.item()
            print(f"Epoch {epoch+1}\tDiscrimnative Loss: {running_discriminator_loss / len(test_loader)} \tGenerative loss: {running_generator_loss / len(test_loader)}")
            test_outputs.append(op_li)
            view_images(op_li, epoch)
        dloss.append(running_discriminator_loss / len(test_loader))
        gloss.append(running_generator_loss / len(test_loader))

    return test_outputs, dloss, gloss
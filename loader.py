from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def trainLoader(batch_size):
    '''
    This function loads the training MNIST dataset
    :param batch_size: batch size to be used during training
    :return: DataLoader object of the train set
    '''
    transform = transforms.ToTensor()
    train_data = datasets.CIFAR10(
        root="./torch_datasets",
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader


def testLoader(batch_size):
    '''
    This function loads the testing MNIST dataset
    :param batch_size: batch size to be used during testing
    :return: DataLoader object of the test set
    '''
    transform = transforms.ToTensor()
    test_data = datasets.CIFAR10(
        root="./torch_datasets",
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_loader

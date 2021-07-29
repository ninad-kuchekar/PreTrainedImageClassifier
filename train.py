import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.utils.data

from collections import OrderedDict
from PIL import Image
import torchvision.models as models
import argparse

argumentparser = argparse.ArgumentParser(description='train.py')
# Command Line arguments

argumentparser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
argumentparser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
argumentparser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
argumentparser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
argumentparser.add_argument('--dropout', dest="dropout", action="store", default=0.2)
argumentparser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
argumentparser.add_argument('--arch', dest="arch", action="store", default="vgg11", type=str)
argumentparser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

parser = argumentparser.parse_args()
data_path = parser.data_dir
paths = parser.save_dir
lr = parser.learning_rate
model_name = parser.arch
dropout = parser.dropout
hidden_layer = parser.hidden_units
power = parser.gpu
epochs = parser.epochs


def load_datasets():
    data_directory = data_path
    train_directory = data_directory + '/train'
    validation_directory = data_directory + '/valid'
    test_directory = data_directory + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_directory, transform=train_transforms)
    validation_data = datasets.ImageFolder(validation_directory, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_directory, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)

    return trainloader, validationloader, testloader


trainloader, validationloader, testloader = load_datasets()


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def network_setup():
    # TODO: Build and train your network
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    if model_name == 'vgg13':
        model = models.vgg13(pretrained=True)
    if model_name == 'vgg19':
        model = models.vgg19(pretrained=True)

    # Turning off gradient for my VGG19 model
    for param in model.parameters():
        param.requires_grad = False

    # Define new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_layer)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=dropout)),
        ('fc2', nn.Linear(hidden_layer, hidden_layer / 2)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=dropout)),
        ('fc3', nn.Linear(hidden_layer / 2, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.fc = classifier
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    model.to(device)
    return model, device, criterion, optimizer


model, device, criterion, optimizer = network_setup()


# Training network and validating with validation set
def train_network():
    # Training  the network

    epochs = 30
    steps = 0

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # validation loop
        else:
            test_loss = 0
            accuracy = 0

            with torch.no_grad():
                model.eval()
                for images, labels in validationloader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    test_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()

            train_losses.append(running_loss / len(trainloader))
            test_losses.append(test_loss / len(validationloader))

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(test_loss / len(validationloader)),
                  "Test Accuracy: {:.3f}".format(accuracy / len(validationloader)))
    return True


train_network()


def testing(accuracy=None):
    epochs = 10
    print_every = 5
    steps = 0
    running_loss = 0
    train_losses, test_losses = [], []
    trainer_size = len(trainloader)
    tester_size = len(testloader)

    # Training loop
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Testing the network - turning off dropout
            if steps % print_every == 0:
                model.eval()

                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for images, labels in testloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()

                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                train_losses.append(running_loss / trainer_size)
                test_losses.append(test_loss / tester_size)

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / tester_size:.3f}.. "
                      f"Test accuracy: {accuracy / tester_size:.3f}")

                running_loss = 0
                model.train()
    return True


testing()


def calculate_accuracy():
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available else "cpu" )
    with torch.no_grad ():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max (outputs.data,1)
            total += labels.size (0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: %d %%' % (100 * correct / total))


calculate_accuracy()


# TODO: Save the checkpoint
def save_checkpoint(save_dir):
    model.to('cpu')
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epochs': epochs}
    torch.save(checkpoint, save_dir + '/checkpoint.pth')


save_checkpoint(paths)


# Load the checkpoint
def load_checkpoint(save_dir):
    checkpoint = torch.load(save_dir + '/checkpoint.pth')
    model = models.vgg19(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    model.hidden_layers = checkpoint['hidden_layers']

    for param in model.parameters():
        param.requires_grad = False

    return model


model = load_checkpoint(paths)


def process_image(image):
    # TODO: Process a PIL image for use in a PyTorch model
    # Using PIL to open image
    image_pil = Image.open(image)

    # Creating transformer settings for the image
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Normalizing image with the transformer
    normalized_image = transformer(image_pil)

    # convert normalized image to numpy array
    normalized_image = np.array(normalized_image)

    # Transposing to change colour dimension to first dimension
    normalized_image = normalized_image.transpose(1, 2, 0)

    # print("Processed normalized image shape:: ",normalized_image.shape)
    return normalized_image


# image = (data_directory + '/test' + '/12/' + 'image_04077.jpg')
# image = process_image(image)


# print(image.shape)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    image = np.array(image)
    image = image.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


#image_path = 'flowers/train/10/image_07086.jpg'
#img = process_image(image_path)
#imshow(img)

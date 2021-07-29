import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import train

#Command Line Arguments

argumentparser = argparse.ArgumentParser()
argumentparser.add_argument('input_img', default='/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
argumentparser.add_argument('checkpoint', default='checkpoint.pth', nargs='*', action="store",type = str)
argumentparser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
argumentparser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
argumentparser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

parser = argumentparser.parse_args()
path_images = parser.input_img
number_of_outputs = parser.top_k
power = parser.gpu
input_img = parser.input_img
path = parser.checkpoint
category_names = parser.category_names


training_loader, testing_loader, validation_loader = train.load_datasets()


model = train.load_checkpoint(path)


with open(category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)


def predict(image_path, model, number_of_outputs=5):

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model.to(device)

    image = train.process_image(image_path)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = torch.unsqueeze(image, dim=0)

    with torch.no_grad():
        output = model.forward(image)

    probability = torch.nn.functional.softmax(output[0], dim=0)

    probabilities, classes = probability.topk(number_of_outputs)

    probabilities = probabilities.numpy()
    classes = classes.numpy()

    probabilities = probabilities.tolist()
    classes = classes.tolist()

    mapping = {val: key for key, val in
               model.class_to_idx.items()
               }

    classes = [mapping[item] for item in classes]
    classes = np.array(classes)
    class_names = [cat_to_name[str(item+1)] for item in classes]

    return probabilities, class_names


probabilities, classes = predict(path_images, model, number_of_outputs)
print("The probabilities for Flowers: {} are \n {}".format(classes, probabilities))



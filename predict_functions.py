import numpy as np
import torch
from torchvision import models
from utils import process_image
import json
from torch import nn, optim
from collections import OrderedDict




#loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == "vgg16":
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch']  == "vgg11":
        model = models.vgg11(pretrained=True)
    else:
        print("please choose either vgg16 or vgg11")
    
    for param in model.parameters():
        param.requires_grad = False
        
    epoch = checkpoint['epoch']
    dropout = checkpoint['dropout']
    hidden_units = checkpoint['hidden_units']
    learning_rate = checkpoint['learning_rate']
    
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, hidden_units)),
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(dropout)),
                                ('fc2', nn.Linear(hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
   
    model.load_state_dict(checkpoint['model_state_dict'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model



def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    model.eval()
    image = process_image(image_path, gpu)
    image = image.unsqueeze_(0)
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        output= model.forward(image)

    output = output.to(device)
    
    probabilities = torch.exp(output).data   
    
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    label = [idx_to_class[idx] for idx in index]
    
    return prob, label


def sanity_check(prob, classes, cat_to_name_path):
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
        
    max_index = np.argmax(prob)
    max_probability = prob[max_index]
    label = classes[max_index]

    labels = []
    for cl in classes:
        
        labels.append(cat_to_name[cl])
        
    return labels
    
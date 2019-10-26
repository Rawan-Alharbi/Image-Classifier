import numpy as np
import torch
from torchvision import models
from utils import process_image
import json



#loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']


    return model



def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if gpu==True else "cpu")

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
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return prob, label


def sanity_check(prob, classes):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    max_index = np.argmax(prob)
    max_probability = prob[max_index]
    label = classes[max_index]

    labels = []
    for cl in classes:
        
        labels.append(cat_to_name[cl])
        
    return labels
    
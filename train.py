import argparse 
from utils import load_data 
from train_functions import build_model, train_model, test_model, save_checkpoint


    

def get_args():
    parser = argparse.ArgumentParser(description= "Train neural network to classify images")
    parser.add_argument("data_dir", help = "path to the data", type = str)
    parser.add_argument("--save_dir", help = "directory to save the data", default= "checkpoint.pth", type = str)
    parser.add_argument("--arch", help = "neural network architecture", choices=['vgg16', 'vgg1'], default= "vgg16")
    parser.add_argument("--learning_rate", "-lr", help = "learning rate of the network", default=0.001, type = float)
    parser.add_argument("--hidden_units", help = "number of hidden units in network", default=512, type = int)
    parser.add_argument("--epochs", help = "number of epochs", default=10, type = int)
    parser.add_argument("--gpu", help = "Use GPU to train the network", action="store_true")
    
    args = parser.parse_args()


    return args



def main():
    args = get_args()
    trainloader, validloader, testloader, train_dataset = load_data(args.data_dir)
    model, optimizer, criterion = build_model(args.arch, args.learning_rate, args.hidden_units, args.gpu)
    train_model(model, optimizer, criterion, trainloader, validloader, args.epochs, args.gpu)
    test_model(model, testloader, args.gpu)
    save_checkpoint(model, optimizer, train_dataset, args.save_dir, args.epochs, args.arch, args.hidden_units, args.learning_rate)

    
   
    
if __name__ == "__main__" :
    main()
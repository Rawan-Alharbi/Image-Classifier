from torchvision import models
from torch import nn, optim
import torch
from collections import OrderedDict



def build_model(archs, learning_rate, hidden_units, gpu):
    
    if archs == "vgg16":
        model = models.vgg16(pretrained=True)
    elif archs == "vgg11":
        model = models.vgg11(pretrained=True)
    else:
        print("please choose either vgg16 or vgg11")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, hidden_units)),
                                ('relu', nn.ReLU()),
                                ('dropout', nn.Dropout(0.2)),
                                ('fc2', nn.Linear(hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    return model, optimizer, criterion


def train_model(model, optimizer, criterion, trainloader, validloader, epochs, gpu):
    steps = 0
    running_loss = 0
    print_every = 40
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)


    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

                

def test_model(model, testloader, gpu):
    correct = 0
    total = 0
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)



    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    
    
def save_checkpoint(model, optimizer, train_dataset, save_dir, epochs, arch, hidden_units, lr, dropout=0.2):
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'model_state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': epochs,
                  'arch': arch,
                  'learning_rate': lr,
                  'dropout': dropout,
                  'hidden_units': hidden_units}

    torch.save(checkpoint, save_dir)
 
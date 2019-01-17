# PROGRAMMER: Mertens Moreno, Carlos Edgar
# DATE CREATED: 05/19/2018                                  
# REVISED DATE: 06/09/2018  - improved according to reviewer's feedback
# PURPOSE: Train a Neural Network Image Classifier with PyTorch to predict a flower image. 
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --save_dir <directory to save the checkpoint> --arch <model>
#             --learning_rate <use for the optimizer> --hidden_units <to be update the architecture>
#               --epochs <epochs to train> --gpu <to use GPU for training and testing>
#   Example call:
#    >>python train.py --save_dir save_directory --arch "vgg16" --learning_rate 0.001 --hidden_units 512 
#       --epochs 8 --gpu

# Imports
import torch
import numpy as np
import argparse
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from classifier import ModelClassifier


def get_args():
    """
    Function to retrieve and parse the command line arguments, 
    then to return these arguments as an ArgumentParser object.
    Parameters:
     None.
    Returns:
     parser.parse_args(): inputed or default argument objects. 
    """      
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='flowers', help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='save_directory', help='directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='densenet121', help='model architecture densenet121 or vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='hyperparameters for learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='hyperparameters for hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='hyperparameters for epochs')
    parser.add_argument('--gpu', action='store_true', help='hyperparameters for GPU')

    return parser.parse_args()


# Call function to get command line arguments
in_arg = get_args()       


def prep_data(train_path, valid_path, test_path):
    """
    Function to prepare the data for training and testing. Use transforms to normalized 
    the data and ImageFolder to retrieve the data.
    Parameters:
     train_path, valid_path and test_path: data sets to be prepare.
    Returns:
     train_loader, valid_loader, test_loader: Sets ready to be use by the model.
     train_datasets: to be used to classify the labels.
    """
    # Define transforms for training, validation, and testing sets
    # For training sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                           transforms.RandomResizedCrop(224), 
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.RandomRotation(45),
                                           transforms.ToTensor(), 
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # For validation and testing sets
    data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_path, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_path, transform=data_transforms)
    test_datasets = datasets.ImageFolder(test_path, transform=data_transforms)
    
    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    
    return train_loader, valid_loader, test_loader, train_datasets


# Define set data directories' path
data_dir = in_arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Call function to load and prepare the data
train_loader, valid_loader, test_loader, train_datasets = prep_data(train_dir, valid_dir, test_dir)


# Build network architecture
def model_network(arch_input):
    """
    Function to create the network. Download a pre-trained CNN and update the classifiers.
    Parameters:
     arch_input: String, model architecture inputed by the user
    Returns:
     model: Neural Network Model
    """
    # Load the pre-trained network DenseNet or VGG
    if arch_input == "densenet121":
        model = models.densenet121(pretrained=True)
        in_features = 1024
    elif arch_input == "vgg16":
        model = models.vgg16(pretrained=True)
        in_features = 25088
        
    # # Freeze feature parameters in order to update our classifier architecture
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Update classifiers calling ModelClassifier function
    model.classifier = ModelClassifier(in_features, in_arg.hidden_units, in_arg.hidden_units//2)
    
    return model, in_features


# Call function to build the Neural Network and retrieve the in_features
model, in_features = model_network(in_arg.arch)

# Initiate our criterion, optimizer and epochs
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
epochs = in_arg.epochs

# Initiate our variables to start the iteration on the datasets
steps = 0
running_loss = 0
print_every = 100

# Initiate iteration to train the network according to the epochs defined
print('Initiate training...\n   ...tracking validation on the on the network:\n')
for e in range(in_arg.epochs):
    
    # Model in training mode, dropout is on
    model.train()
    
    # Iterate through the training datasets
    for images, labels in iter(train_loader):
        steps += 1        
        
        # Clear optimizer
        optimizer.zero_grad()        
        
        # Check for GPU, if available bring our model and variables up for calculations
        if in_arg.gpu and torch.cuda.is_available():
            model.cuda()
            images = images.cuda()
            labels = labels.cuda()       
        
        # Forward the data through the network and backprop
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        # Track the running loss on the model
        running_loss += loss.item()
        
        # Validating (Inference) the training with different datasets very x steps
        if steps % print_every == 0:
            
            # Model in inference mode, dropout is off
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
            
                # Set variables for validation
                accuracy = 0
                valid_loss = 0
            
                # Initiate iteration on the validation datasets
                for images, labels in iter(valid_loader):               
                
                    # Check for GPU
                    if in_arg.gpu and torch.cuda.is_available():
                        model.cuda()
                        images = images.cuda()  #inputs = inputs.cuda()
                        labels = labels.cuda()  #labels = labels.cuda()
                
                    # Forward the data through the network 
                    output = model.forward(images)
                    valid_loss += criterion(output, labels).item()               
                
                    # Calculate prob accuracy with exponential since output uses log-softmax
                    ps = torch.exp(output).data
                
                    # Highest probability class compare with true label
                    equality = (labels.data == ps.max(1)[1])
                
                    # Accuracy is the mean of correct predictions and all predictions
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
            
                # Print the validation test
                print('Epoch: {}/{}.. '.format(e + 1, epochs), 
                      'Training Loss: {:.3f}.. '.format(running_loss/print_every), 
                      'Validation Loss: {:.3f}.. '.format(valid_loss/len(valid_loader)), 
                      'Validation Accuracy: {:.3f}'.format(accuracy/len(valid_loader)))
            
                # Clear running loss for next step
                running_loss = 0        
        
            # Model back on training
            model.train()
print('Training finished...\n   ...Initiating testing on new dataset:\n')


def test_model(data_path):
    """
    Function to test the trained model with a new set of data
    Parameters:
     data_path: String, directory's path to the dataset 
    Returns:
     float, prints out the accuracy of the network
    """
    # # Model in inference mode, dropout is off
    model.eval()
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():

        # Set variables 
        accuracy = 0
        test_loss = 0

        # Initite iterstion on the datasets
        for images, labels in iter(data_path):

            # Check for GPU
            if in_arg.gpu and torch.cuda.is_available():
                model.cuda()
                images = images.cuda()
                labels = labels.cuda()
        
            # Feedforward inputs through the network and track the loss
            output = model.forward(images)
            test_loss += criterion(output, labels).item()                
        
            # Calculate the accuracy 
            ps = torch.exp(output).data
        
            # Compare with true label
            equality = (labels.data == ps.max(1)[1])
        
            # Calculate the mean to define accuracy
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    # Return the accuracy of the datasets            
    return print('Test Accuracy on the test datasets: {}'.format(accuracy/len(test_loader)))


# Call function to test the model with new data
test_model(test_loader)


def save_checkpoint():
    '''
    Function to save the hyperparameters used to train the neural network.
    Parameters:
    None
    Returns:
     String, prints out the path where the checkpoints have been saved.      
    '''
    checkpoint = {'arch': in_arg.arch,
                  'in_features': in_features,
                  'out_features': 102,
                  'hidden_units': in_arg.hidden_units,
                  'hidden_units2': in_arg.hidden_units//2,
                  'state_dict': model.state_dict(),
                  'criterion': criterion, 
                  'optimizer': optimizer.state_dict(), 
                  'epochs': in_arg.epochs, 
                  'model.class_to_idx': train_datasets.class_to_idx}

    torch.save(checkpoint, in_arg.save_dir + '/checkpoint.pth')
    
    return print('\nHyperparameters has been saved at : ', in_arg.save_dir, '/checkpoint.pth')


# Call function to save checkpoint
save_checkpoint()
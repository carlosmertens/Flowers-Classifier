# PROGRAMMER: Mertens Moreno, Carlos Edgar
# DATE CREATED: 05/19/2018                                  
# REVISED DATE: 06/10/2018  - improved according to reviewer's feedback
# PURPOSE: Predict a image of a flower using the trained network checkpoint.  
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py --input <path to a flower image> --checkpoint <checkpoint file to load 
#               pre-trained model>
#             --category_names <json file with the labels> --top_k <number of top predictions>
#               --gpu <to use GPU for training and testing>
#   Example call:
#    >>python train.py --input flowers/test/100/image_07896.jpg --checkpoint save_directory/checkpoint.pth 
#                --gpu

# Imports
import matplotlib.pyplot as plt

import torch
import numpy as np
from torchvision import models
from PIL import Image
from classifier import ModelClassifier
import json
import argparse


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
    
    parser.add_argument('--input', type=str, default='flowers/test/100/image_07896.jpg', 
                        help="path to folder of images")
    parser.add_argument('--checkpoint', type=str, default='save_directory/checkpoint.pth', 
                        help='file to load the checkpoint')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', 
                        help='file to map the real names')
    parser.add_argument('--top_k', type=int, default=3, help='top classes predicted to return')
    parser.add_argument('--gpu', action='store_true', help='hyperparameters for GPU')

    return parser.parse_args()


# Call function to get command line arguments
in_arg = get_args()


def load_checkpoint(filepath):
    """
    Function to load the checkpoint and to rebuild the trained network for prediction.
    Parameters:
     filepath: Path to the checkpoint file.
    Returns: 
     model: Pre-trained model loaded and classifiers updated to predict image.
     criterion: The criterion used to pre-trained the network.
     optimizer: The optimozer used to pre-trained the network.
     epochs: The epochs used to pre-trained the network.
     class_idx: Classes to index saved on trining
    """    
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    
    # Load the network
    if checkpoint['arch'] == "densenet121":
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == "vgg16":
        model = models.vgg16(pretrained=True) 
    
    model.classifier = ModelClassifier(checkpoint['in_features'], checkpoint['hidden_units'], 
                                       checkpoint['hidden_units2'])
    
    model.load_state_dict(checkpoint['state_dict'])    
    criterion = checkpoint['criterion']    
    optimizer = checkpoint['optimizer']    
    epochs = checkpoint['epochs']    
    class_idx = checkpoint['model.class_to_idx']
    
    return model, criterion, optimizer, epochs, class_idx


# Call function to load the model and its hyperparameters
model, criterion, optimizer, epochs, model.class_to_idx = load_checkpoint(in_arg.checkpoint)


def process_image(image):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    Parameters:
     image: Image to be processed
    Returns: 
     ing: an Numpy array to pass it to the model
    '''
    # Open image
    img = Image.open(image)
    
    # Resize image, keep aspect radio, keep 256 pixels on the shortest side
    if img.size[0] >= img.size[1]: 
        img.thumbnail((1024,256))
    else: 
        img.thumbnail((256,1024))

    # Set variables, , calculate box dimension, crop image
    width = img.size[0] / 2
    height = img.size[1] / 2
    box = (width - 112, height - 112, width + 112, height + 112)
    img = img.crop(box)
    
    # Set mean and standard deviations, 
    mean = [0.485, 0.456, 0.406] 
    stdv = [0.229, 0.224, 0.225]
    
    # Convert color chanel from 0-255 to 0-1
    np_image = np.array(img)
    img = np_image/255
    
    # Normalize and re-order dimension
    img = (img-mean)/stdv
    img = img.transpose((2,0,1))
    
    return img


# Called function to load the image and process it to be predicted
test_image = process_image(in_arg.input)


def predict(image, model, topk=in_arg.top_k):
    ''' 
    Function to predict the class (or classes) of an image using a trained deep learning model.
    Parameters:
     image: Image already prepared to pass through the network
     model: Model chosen to predict the image
     topk: Number of top predictions chosen
    Returns:
     top_probs: Top probabilities in porcentage
     top_classes: Top classes indexed on the network
    '''
    model.eval()
    
    # Set variables 
    img = np.expand_dims(image, axis=0)
    inputs = torch.from_numpy(img).float()
    
    with torch.no_grad():
        # Check for GPU
        if in_arg.gpu and torch.cuda.is_available():
            model.cuda()
            inputs = inputs.cuda()

        # Pass input through the network
        output = model.forward(inputs)
                
        # Get probabilities
        ps = torch.exp(output).data.topk(topk)
        
        # Unpacking the Tensor 
        probs, classes = ps
    
        # Convert to array 
        probs = probs.data[0].cpu().numpy()
        top_probs = [round(each * 1, 3) for each in probs]
        
        idx_to_class = dict(zip(model.class_to_idx.values(), model.class_to_idx.keys()))
        classes = classes.data[0].cpu().numpy()
        top_classes = [idx_to_class[each] for each in classes] 
        
    return top_probs, top_classes


# Call function to predict the image
probs_k, classes_k = predict(test_image, model)

# Load json file with the names to categorize
with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Label the top classes 
classes_k_name = [cat_to_name[each] for each in classes_k]

# Display results
print('\n\n*** PREDICTIONS ***\n')

for i in range(0, in_arg.top_k):
    name = classes_k_name[i].title()
    porc = round(probs_k[i]*100, )
    print(name, porc,"%")

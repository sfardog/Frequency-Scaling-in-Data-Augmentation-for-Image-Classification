import torch

from .get_functions import get_load_path

def load_model(args, eecp, model_dir='./model_save') :
    load_model_path = get_load_path(args, eecp, model_dir)

    print("Your model is loaded from {}.".format(load_model_path))
    checkpoint = torch.load(load_model_path)
    print(".pth keys() =  {}.".format(checkpoint.keys()))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    epochs = checkpoint['epochs']
    model_name = checkpoint['model_name']

    return model, epochs, model_name
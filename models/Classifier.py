import sys

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Classifier(nn.Module) :
    def __init__(self, model_name, num_classes):
        super(Classifier, self).__init__()

        self.model = get_model(model_name, num_classes)
        self.model.apply(self.weights_init)
        print("Model weight initialization complete!")

    def forward(self, x):
        out = self.model(x)
        # out = F.softmax(out)

        return out

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.05)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.05)
            nn.init.constant_(m.bias.data, 0)

def get_model(model_name, num_classes) :
    if model_name == 'wide_resnet' :
        from models.wideresnet import wideresnet
        model = wideresnet(num_classes, 3)

    elif model_name == 'resnext50_32x4d' :
        from models.resnext import resnext50
        model = resnext50(num_classes, 3)

    elif model_name == 'resnet101' :
        from models.resnet import resnet101
        model = resnet101(num_classes, 3)

    elif model_name == 'densenet121' :
        from models.densenet import densenet121
        model = densenet121(num_classes, 3)

    else :
        print('wrong model choice')
        sys.exit()

    return model
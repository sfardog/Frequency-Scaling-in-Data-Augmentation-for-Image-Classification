import torch.nn as nn

from .Extract.ExtractModel import ExtractModel
from .Emphasize import Emphasize

class EECPmodule(nn.Module) :
    def __init__(self, args, device):
        super(EECPmodule, self).__init__()

        self.args = args
        self.extract = ExtractModel(args, device)
        self.emphasize = Emphasize(device, args, self.extract).to(device)

    def forward(self, x):
        pattern_importance = self.extract(x)
        new, pattern_importance = self.emphasize(x, pattern_importance)

        # if not self.args.parallel : return new, pattern_importance
        # else:return new, pattern_importance, self.extract.clustered_idx, self.extract.X, self.extract.labels
        # print(self.extract.clustered_idx)
        return new, pattern_importance#, self.extract.clustered_idx, self.extract.X, self.extract.labels

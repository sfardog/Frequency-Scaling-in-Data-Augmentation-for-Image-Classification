import torch.nn as nn

class SEblock(nn.Module) :
    def __init__(self, c, r=16):
        super(SEblock, self).__init__()
        self.channels = c
        self.reduction = r
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False), # nn.Linear(3072, 3072//16, bias=False)
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel = x.size()[0], x.size()[1]
        se = self.squeeze(x).view(batch, channel)
        se = self.excitation(se).view(batch, channel, 1, 1)

        return x * se.expand_as(x)
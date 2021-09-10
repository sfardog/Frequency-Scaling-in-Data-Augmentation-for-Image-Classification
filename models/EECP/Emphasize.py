import torch
import torch.nn as nn
import torch.fft as fft

class Emphasize(nn.Module):
    def __init__(self, device, args, extract):
        super(Emphasize, self).__init__()

        self.device = device
        self.extract = extract
        self.args = args

    def forward(self, x, pattern_importance):
        x_fft = fft.fftshift(fft.fft2(x))
        mask = torch.ones((x_fft.size(0), x_fft.size(1), x_fft.size(2))).to(self.device)
        for i, (idxx, idxy) in enumerate(self.extract.clustered_idx) :
            for j in range(len(pattern_importance)) :
                mask[j, idxx, idxy] = pattern_importance[j, i]*self.args.weight_factor

        # import matplotlib.pyplot as plt
        # plt.imshow(mask[0].cpu().detach().numpy(), cmap='gray')
        # plt.show()

        x_reject = x_fft * mask
        x_ifft = torch.abs(fft.ifft2(fft.ifftshift(x_reject)))

        if self.args.combination == 'only_ifft' :
            x_new = x_ifft.unsqueeze(1).repeat(1, 3, 1, 1)
        else :
            x_diff = torch.abs(x_ifft - x)
            x_new = torch.zeros((x.size(0), 3, x.size(1), x.size(2)))

            x_new[:, 0, :, :] = x
            x_new[:, 1, :, :] = x_ifft
            x_new[:, 2, :, :] = x_diff

        return x_new, pattern_importance
import torch
import torch.nn as nn

from utils import *

class Discriminator(nn.Module):
    def __init__(self, input_shape, device):
        super(Discriminator, self).__init__()
        self.device = device

        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** 5, width // 2 ** 5)
        stacked_channels = channels

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.shared_layers = nn.Sequential(
            *discriminator_block(stacked_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512)
        )

        self.discriminator_layers = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )


    def forward(self, stacked_frame=None, task='pose'):
        shared_output = self.shared_layers(stacked_frame)
        discriminator_output = self.discriminator_layers(shared_output)
        return discriminator_output


if __name__ == "__main__":
    disc = Discriminator(input_shape=(6, 128, 128), device='cpu').to('cpu')

    test = torch.randn(12, 3, 128, 128).to('cpu')

    test_test = torch.cat([test, test], dim=1)

    disc_output = disc(test_test)

    print(f"disc.shape: {disc_output.shape}")

    disc_output = (disc.output_shape[0], 2 * disc.output_shape[1], 2 * disc.output_shape[2])
    print(disc_output)
from torch import nn


class CustomAutoEncoderV1(nn.Module):
    def __init__(self, im_channel: int = 1):
        super(CustomAutoEncoderV1, self).__init__()
        self._im_c = im_channel
        self._encoder = nn.Sequential(
            nn.Conv2d(self._im_c, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, self._im_c, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        x = self._encoder(input_tensor)
        x = self._decoder(x)
        return x




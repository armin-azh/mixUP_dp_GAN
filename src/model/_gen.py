from torch import nn


class CustomGeneratorV1(nn.Module):
    def __init__(self, gpu_num: int, feature_num: int, z_num: int, output_channel: int) -> None:
        super(CustomGeneratorV1, self).__init__()
        self._gpu_num = gpu_num
        self._num_feature = feature_num
        self._num_z_vec = z_num
        self._output_channel = output_channel
        self._sequence_model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self._num_z_vec,
                               out_channels=self._num_feature * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(self._num_feature * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self._num_feature * 8,
                               out_channels=self._num_feature * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self._num_feature * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self._num_feature * 4,
                               out_channels=self._num_feature * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self._num_feature * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self._num_feature * 2,
                               out_channels=self._num_feature,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self._num_feature),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=self._num_feature,
                               out_channels=self._output_channel,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        return self._sequence_model(input_tensor)

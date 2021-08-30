from torch import nn


class CustomDiscriminatorV1(nn.Module):
    def __init__(self, gpu_num: int, feature_num: int, input_channel: int):
        super(CustomDiscriminatorV1, self).__init__()
        self._input_channel = input_channel
        self._num_feature = feature_num
        self._gpu_num = gpu_num

        self._sequence_model = nn.Sequential(
            nn.Conv2d(in_channels=self._input_channel,
                      out_channels=self._num_feature,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self._input_channel,
                      out_channels=self._input_channel * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self._input_channel * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self._input_channel * 2,
                      out_channels=self._input_channel * 4,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self._input_channel * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=self._input_channel * 8,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid()

        )

    def forward(self, input_tensor):
        return self._sequence_model(input_tensor)

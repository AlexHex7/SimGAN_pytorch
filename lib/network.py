from torch import nn


# class ResnetBlock(nn.Module):
#     def __init__(self, input_features, nb_features=64):
#         super(ResnetBlock, self).__init__()
#         self.convs = nn.Sequential(
#             nn.Conv2d(input_features, nb_features, 3, 1, 1),
#             # nn.BatchNorm2d(nb_features),
#             nn.ReLU(),
#             nn.Conv2d(nb_features, nb_features, 3, 1, 1),
#             # nn.BatchNorm2d(nb_features)
#         )
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         convs = self.convs(x)
#         sum = convs + x
#         output = self.relu(sum)
#         return output
#
#
# class Refiner(nn.Module):
#     def __init__(self, block_num, in_features, nb_features=64):
#         super(Refiner, self).__init__()
#
#         self.conv_1 = nn.Sequential(
#             nn.Conv2d(in_features, nb_features, 3, stride=1, padding=1),
#             # nn.BatchNorm2d(nb_features)
#         )
#
#         blocks = []
#         for i in range(block_num):
#             blocks.append(ResnetBlock(nb_features, nb_features))
#
#         self.resnet_blocks = nn.Sequential(*blocks)
#
#         self.conv_2 = nn.Sequential(
#             nn.Conv2d(nb_features, in_features, 1, 1, 0),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         conv_1 = self.conv_1(x)
#
#         res_block = self.resnet_blocks(conv_1)
#         output = self.conv_2(res_block)
#         return output

# class Discriminator(nn.Module):
#     def __init__(self, input_features):
#         super(Discriminator, self).__init__()
#
#         self.convs = nn.Sequential(
#             nn.Conv2d(input_features, 96, 3, 2, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(96),
#
#             nn.Conv2d(96, 64, 3, 2, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#
#             nn.MaxPool2d(3, 2, 1),
#
#             nn.Conv2d(64, 32, 3, 1, 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#
#             nn.Conv2d(32, 32, 1, 1, 0),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#
#             nn.Conv2d(32, 2, 1, 1, 0),
#             nn.ReLU(),
#             # nn.BatchNorm2d(2),
#             nn.MaxPool2d(3, 2, 1),
#         )
#
#     def forward(self, x):
#         convs = self.convs(x)
#         output = convs.view(convs.size(0), -1, 2)
#         return output


class ResnetBlock(nn.Module):
    def __init__(self, input_features, nb_features=64):
        super(ResnetBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_features, nb_features, 3, 1, 1),
            nn.BatchNorm2d(nb_features),
            nn.LeakyReLU(),
            nn.Conv2d(nb_features, nb_features, 3, 1, 1),
            nn.BatchNorm2d(nb_features)
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        convs = self.convs(x)
        sum = convs + x
        output = self.relu(sum)
        return output


class Refiner(nn.Module):
    def __init__(self, block_num, in_features, nb_features=64):
        super(Refiner, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_features, nb_features, 3, stride=1, padding=1),
            nn.BatchNorm2d(nb_features)
        )

        blocks = []
        for i in range(block_num):
            blocks.append(ResnetBlock(nb_features, nb_features))

        self.resnet_blocks = nn.Sequential(*blocks)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(nb_features, in_features, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, x):
        conv_1 = self.conv_1(x)

        res_block = self.resnet_blocks(conv_1)
        output = self.conv_2(res_block)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_features):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(input_features, 96, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 64, 3, 2, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            nn.AvgPool2d(3, 2, 1),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            nn.AvgPool2d(3, 2, 1),

            nn.Conv2d(32, 2, 1, 1, 0),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(2),
        )

    def forward(self, x):
        convs = self.convs(x)
        output = convs.view(convs.size(0), -1, 2)
        return output



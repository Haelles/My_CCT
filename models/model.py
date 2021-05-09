import torch
import torch.nn as nn
from models.transformer import TransformEncoder


class Tokenization(nn.Module):
    def __init__(self, conv_layers_num=1, input_channel=3, embed_channel=64, out_channel=256, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = out_channel
        self.feature_size = [input_channel]
        for i in range(conv_layers_num - 1):
            self.feature_size.append(embed_channel)
        self.feature_size.append(out_channel)

        self.conv_blocks = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(self.feature_size[i], self.feature_size[i + 1], kernel_size, stride, padding, bias=False),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                ) for i in range(conv_layers_num)
            ]
        )
        self.flatten = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def get_sequence_len(self):
        return self.forward(torch.zeros(1, self.input_channel, 32, 32)).size(1)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x).transpose(2, 3)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, encoder_num=7, embed_dim=256, class_num=10):
        super().__init__()
        self.encoders = nn.Sequential(
            *[TransformEncoder() for i in range(encoder_num)]
        )
        self.sequence_pool = nn.Linear(embed_dim, 1)
        self.soft = nn.Softmax()
        self.mlp_layer = nn.Linear(embed_dim, class_num)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.encoders(x)  # b, n, d
        temp = self.soft((self.sequence_pool(x)).transpose(-2, -1))  # -> b, 1, n
        x = (temp @ x).squeeze(1)  # b, 1, d -> b, d
        x = self.mlp_layer(x)
        return x


class CCT(nn.Module):
    def __init__(self, embed_dim=256, conv_layers_num=1, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.tokenization = Tokenization()
        self.position = nn.Parameter(torch.zeros(1, self.tokenization.get_sequence_len(), embed_dim))
        nn.init.trunc_normal_(self.position, std=0.2)
        self.layer_norm = nn.LayerNorm(embed_dim)
        nn.init.constant_(self.layer_norm.bias, 0)
        nn.init.constant_(self.layer_norm.weight, 1.0)
        self.classifier = TransformerClassifier()

    def forward(self, x):
        x = self.tokenization(x)
        x += self.position
        x = self.layer_norm(x)  # TODO need to check
        x = self.classifier(x)  # b, class_num
        return x

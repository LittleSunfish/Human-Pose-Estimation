import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):

    def __init__(self, in_channels, reduced_channels, out_channels, stride=1, starting=False):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != 4*reduced_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, 4*reduced_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(4*reduced_channels)
            )

        self.final_relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out) + self.shortcut(x)
        out = self.final_relu(out)
        return out

class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.num_blocks = [3,4,6,3]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = self._make_layer(in_channels=64, reduced_channels=64, out_channels=256, num_blocks=self.num_blocks[0], stride=1)
        self.conv3 = self._make_layer(in_channels=256, reduced_channels=128, out_channels=512, num_blocks=self.num_blocks[1], stride=1)
        self.conv4 = self._make_layer(in_channels=512, reduced_channels=256, out_channels=1024, num_blocks=self.num_blocks[2], stride=1)
        self.conv5 = self._make_layer(in_channels=1024, reduced_channels=512, out_channels=2048, num_blocks=self.num_blocks[3], stride=1)
        
        self.avg = nn.AvgPool2d(kernel_size=4)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=17, kernel_size=1)
        )

    def _make_layer(self, in_channels, reduced_channels, out_channels, num_blocks, stride=1):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)

        for i, s in enumerate(strides):
            bottleneck_layer = Bottleneck(in_channels, reduced_channels, out_channels, s)
            layers.append(bottleneck_layer)
            in_channels = reduced_channels *4
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)   
        out = self.conv2(out)   # output: 256*64*48
        out = self.conv3(out)   # output: 512*32*24
        out = self.conv4(out)
        out = self.conv5(out)   # output: 16*12
        #out = self.avg(out)
        #out = out.view(out.size(0), -1)

        ## the input of last layer shoul be in size 64x48 before
        out = self.final(out)
        return out

def main():
    from torchsummary import summary

    resnet = ResNet50().cuda()
    summary(resnet, input_size=(3,256,192))
    
if __name__ == '__main__':
    main()

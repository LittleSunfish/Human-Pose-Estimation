import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, planes*expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes*expansion),
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or inplanes != expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expansion*planes)
            )

        self.final_relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out) + self.shortcut(x)
        out = self.final_relu(out)
        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, num_blocks, num_inchannels, num_channels):
        super(HighResolutionModule, self).__init__()

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches

        self.branches = self.make_branches(num_branches, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)
        

    def _make_one_branch(self, branch_index, num_blocks, num_channels, stride=1):
        layers = []
        # one fundamental layer
        layers.append(
            Bottleneck(self.num_inchannels[branch_index], num_channels[branch_index], stride=1)
        )

        # layers according to num_blocks[branch_index]
        for i in range(num_blocks[branch_index]):
            layers.append(
                Bottleneck(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        # implementation of exchange unit in paper
        if self.num_branches == 1: # nothing to fuse
            return None 

        num_inchannels = self.num_inchannels
        fuse_layers = []
        for cur in range(self.num_branches):
            fuse_layer = []
            for prev in range(num_branches):
                if prev > cur:
                    # upsample
                    upsample = nn.Sequential(
                        nn.Conv2d(num_inchannels[prev], num_inchannels[cur], size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(num_inchannels[cur]),
                        nn.Upsample(scale_factor=2**(prev-cur),mode='nearest')
                    )
                    fuse_layer.append(upsample)
                elif prev == cur:
                    fuse_layer.append(None)
                else: # prev < cur
                    downsamples = []
                    prev_change = prev
                    while cur - prev_change > 1:
                        downsample = nn.Sequential(
                            nn.Conv2d(num_inchannels[prev], num_inchannels[cur], size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(num_inchannels[cur])
                        )
                        downsamples.append(downsample)
                        prev_change -= 1
                        
                    final_downsample = nn.Sequential(
                        nn.Conv2d(num_inchannels[prev], num_inchannels[cur], size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(num_inchannels[cur]),
                        nn.ReLU(True)
                    )
                    downsamples.append(final_downsample)
                    fuse_layer.append(nn.Sequential(*downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))    
        
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        pass 



class PoseHRNet(nn.Module):

    def __init__(self):
        super(PoseHRNet, self).__init__()

        # stem : 2 strided convolutions decreasing the resolution 

        # main body : outputting the feature maps with the same resolution 

        # regressor: estimating the heatmaps where the keypoints are chosen and tranformed to the full resolution
        
        pass

    def _make_transition_layer():
        pass

    def _make_layer():
        pass

    def _make_stage():
        pass

    def forward(self, x):
        pass

    # to levergage pretrained model
    def init_weights():
        pass

def main():
    from torchsummary import summary

    hrnet = HRNet().cuda()
    summary(hrnet, input_size=(3,256,192))
    
if __name__ == '__main__':
    main()

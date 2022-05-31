import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernal_size=3, stride=stride, padding=1, bais=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, plane, kernal_size=3),
            nn.BatchNorm2d(planes)
        )
        self.shortcut = nn.Sequantial()
        if stride != 1 or inplanes != expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expansion*planes)
            )
        self.final_relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out) + self.shortcut(x)
        out = self.final_relu(out)
        return out 

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
                        nn.Conv2d(num_inchannels[prev], num_inchannels[prev], size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(num_inchannels[prev]),
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
                            nn.BatchNorm2d(num_inchannels[cur]),
                            nn.ReLU(True)
                        )
                        downsamples.append(downsample)
                        prev_change -= 1
                        
                    final_downsample = nn.Sequential(
                        nn.Conv2d(num_inchannels[prev], num_inchannels[cur], size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(num_inchannels[cur])
                    )
                    downsamples.append(final_downsample)
                    fuse_layer.append(nn.Sequential(*downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))    
        
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        x_fuse = []
        
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse



class PoseHRNet(nn.Module):
    # follows w32_256x256_adam_lr1e-3

    def __init__(self):
        super(PoseHRNet, self).__init__()

        ## stem : 2 strided convolutions decreasing the resolution 
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, size=3, stride=2, padding=1, bias=False),
            nn.BatchNormd(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        ## main body : outputting the feature maps with the same resolution 
        # stage 2
        # num_modules(exchange block): 1, num_branches: 2, num_blocks: 4,4 num_channels: 32, 64
        self.transition1 = self._make_transition_layer()
        self.stage2 = self._make_stage()

        # stage 3
        # num_modules: 4, num_branches: 3, num_blocks: 4,4,4, num_channels: 32,64,128
        self.transition2 = self._make_transition_layer()
        self.stage3 = self._make_stage()

        # stage 4
        # num_modules: 3, num_branches: 4, num_blocks: 4,4,4,4 num_channels: 32,64,128, 256
        self.transition3 = self._make_transition_layer()
        self.stage4 = self._make_stage()

        ## regressor: estimating the heatmaps where the keypoints are chosen and tranformed to the full resolution
        self.final_layer = nn.Conv2d()

        pass

    def _make_transition_layer():
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range()

        return nn.ModuleList(transition_layers)
        

    def _make_layer(self, block, ):
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride))

        self.inplanes = planes * block.expansion
        for i in range(1, block):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)
        

    def _make_stage():
        # use HighResolutionModules

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

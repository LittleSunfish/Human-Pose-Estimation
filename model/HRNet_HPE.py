import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

monitor = 0

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3),
            nn.BatchNorm2d(planes)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
            nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes*self.expansion)
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.final_relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out) + self.shortcut(x)
        out = self.final_relu(out)
        return out

class HighResolutionModule(nn.Module):
    def __init__(self, block, num_branches, num_blocks, num_inchannels, num_channels, multi_scale_output):
        super(HighResolutionModule, self).__init__()
        if monitor ==1:
            print("HighResolutionModule, __init__")

        self.num_inchannels = num_inchannels
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(block, num_branches, num_blocks, num_channels)
        if monitor == 1: print("Module, make_branches done")
        self.fuse_layers = self._make_fuse_layers()
        if monitor == 1: print("Module, fusing done")
        self.relu = nn.ReLU(True)
        

    def _make_one_branch(self, block, branch_index, num_blocks, num_channels, stride=1):
        layers = []
        print("values: ", self.num_inchannels, branch_index, num_channels)
        # one fundamental layer
        layers.append(
            block(self.num_inchannels[branch_index], num_channels[branch_index], stride=1)
        )

        # layers according to num_blocks[branch_index]
        for i in range(num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, block, num_branches, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(block, i, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        # implementation of exchange unit in paper
        if self.num_branches == 1: # nothing to fuse
            return None 

        num_inchannels = self.num_inchannels
        fuse_layers = []
        if self.multi_scale_output == False:
            self.num_branches = 1 
        for cur in range(self.num_branches):
            fuse_layer = []
            for prev in range(self.num_branches):
                if prev > cur:
                    # upsample
                    upsample = nn.Sequential(
                        nn.Conv2d(num_inchannels[prev], num_inchannels[prev], kernel_size=1, stride=1, padding=0, bias=False),
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
                            nn.Conv2d(num_inchannels[prev], num_inchannels[cur], kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(num_inchannels[cur]),
                            nn.ReLU(True)
                        )
                        downsamples.append(downsample)
                        prev_change += 1
                        
                    final_downsample = nn.Sequential(
                        nn.Conv2d(num_inchannels[prev], num_inchannels[cur], kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(num_inchannels[cur])
                    )
                    downsamples.append(final_downsample)
                    fuse_layer.append(nn.Sequential(*downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))    
        
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if monitor ==1 : print("Module, forwad pass begins")
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        if monitor == 1: print("Module, forward, make branches done")
        
        x_fuse = []
        
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        if monitor == 1: print("Module, forward, fusing done")

        return x_fuse



class PoseHRNet(nn.Module):
    # follows w32_256x256_adam_lr1e-3
    block = BasicBlock

    def __init__(self):
        super(PoseHRNet, self).__init__()
        self.inplanes = 64

        ## stem : 2 strided convolutions decreasing the resolution 
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        ## main body : outputting the feature maps with the same resolution 
        # stage 2 -> num_modules(exchange block): 1, num_branches: 2, num_blocks: 4,4 num_channels: 32, 64
        num_channels = [32, 64]
        num_channels = [num_channels[i] * self.block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        if monitor ==1 : print("HRNet, __init__ transition1: \n", self.transition1)
        self.stage2, pre_num_channel = self._make_stage(BasicBlock, 1, 2, [4,4], [32,64], num_channels)
        if monitor ==1 : print("HRNet, __init__, stage2 completed")

        # stage 3 -> num_modules: 4, num_branches: 3, num_blocks: 4,4,4, num_channels: 32,64,128
        num_channels = [32, 64, 128]
        num_channels = [num_channels[i] * self.block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_num_channel, num_channels)
        self.stage3, pre_num_channel = self._make_stage(BasicBlock, 4, 3, [4,4,4], [32,64,128],num_channels)
        if monitor ==1 : print("HRNet, __init__, stage3 completed")

        # stage 4 -> num_modules: 3, num_branches: 4, num_blocks: 4,4,4,4 num_channels: 32,64,128, 256
        num_channels = [32, 64, 128, 256]
        num_channels = [num_channels[i] * self.block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_num_channel, num_channels)
        self.stage4, pre_num_channel = self._make_stage(BasicBlock, 3, 4, [4,4,4,4], [32,64,128,256], num_channels)
        if monitor ==1 : print("HRNet, __init__, stage4 completed")

        ## regressor: estimating the heatmaps where the keypoints are chosen and tranformed to the full resolution
        self.final_layer = nn.Conv2d(pre_num_channel[0], 16, kernel_size=1, stride=1, padding=0)


    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    tran_layer = nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], 
                        nn.ReLU())
                    ) 
                    transition_layers.append(tran_layer)
                else:
                    transition_layers.append(None)
            else:
                conv = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    if j == i-num_branches_pre:
                        outchannels = num_channels_cur_layer[i]
                    else:
                        outchannels = inchannels
                    conv.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU()
                        )
                    )
                transition_layers.append(nn.Sequential(*conv))

        return nn.ModuleList(transition_layers)
        

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        

    def _make_stage(self, block, num_modules, num_branches, num_blocks, num_channels, num_inchannels, multi_scale_output=True):
        # use HighResolutionModules
        if monitor == 1: print("HRNet, _make_stage begins")
        modules = []
        for i in range(num_modules):
            if multi_scale_output == False and i != num_modules -1:
                multi_scale_output = True

            module = HighResolutionModule(block, num_branches, num_blocks, num_inchannels, num_channels, multi_scale_output)
            modules.append(module)
            if monitor == 1: print(f"HRNet, _make_stage {i}th iter module created")
            num_inchannels = modules[-1].get_num_inchannels()
        
        if monitor ==1: print("HRNet, _make_stage just about to return")

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        if monitor ==1 : print("HRNet, forward pass begin")
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.layer1(out)

        tr_out = []
        for i in range(2):
            if monitor ==1: print("HRNet, forward, transition1: \n", self.transition1)
            if self.transition1[i] is not None:

                tr_out.append(self.transition1[i](out))
            else:
                print("NONE")
                tr_out.append(out)
            #tr_out.append(self.transition1[i](out) if self.transition1[i] is not None else out)
        out = self.stage2(tr_out)
        if monitor ==1: print("HRNet, forward, stage2 complemted")

        tr_out = []
        for i in range(3):
            tr_out.append(self.transition2[i](out[-1]) if self.transition2[i] is not None else out[i])
        out = self.stage3(tr_out)

        for i in range(4):
            tr_out.append(self.transition3[i](out[-1]) if self.transition3[i] is not None else out[i])
        out = self.stage4(tr_out)
        if monitor ==1: print("HRNet, forward, stage4 complemted")

        out = self.final_layer(out[0])
        return out

    # to levergage pretrained model
    def init_weights(self, pretrained=''):
        # let every parameter to follow normal distribution
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        # follow pre-trained model if exists 
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def main():
    from torchsummary import summary

    hrnet = PoseHRNet().cuda()
    summary(hrnet, input_size=(3,256,192))
    
if __name__ == '__main__':
    main()

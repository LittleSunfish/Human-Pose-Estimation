from model import HRNet_HPE
from customCOCO import CustomCOCO

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import random
from tqdm import tqdm

PATH = '/home/sojeong/CV/resnet/model_hrnet'

train_coco_json_path = '/home/sojeong/CV/deep-high-resolution-net.pytorch/data/coco/annotations/person_keypoints_train2017.json'
train_coco_img_path  = '/home/sojeong/CV/deep-high-resolution-net.pytorch/data/coco/images/train2017/'
test_coco_json_path = '/home/sojeong/CV/deep-high-resolution-net.pytorch/data/coco/annotations/person_keypoints_val2017.json'
test_coco_img_path = '/home/sojeong/CV/deep-high-resolution-net.pytorch/data/coco/images/val2017'

training_data = CustomCOCO(train_coco_json_path, train_coco_img_path)
#test_data = CustomCOCO(test_coco_json_path, test_coco_img_path)

train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

def reset_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model, n_epoch, loader, optimizer, criterion, device="cpu"):
    model.train()
    for epoch in tqdm(range(n_epoch)):
        running_loss = 0.0
        for i, (images, labels, visibility) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            # print("size of outputs: ", outputs.shape)
            # print("size of labels: ", labels.shape)
            loss = criterion(input=outputs, target=labels)
            loss.mean().backward()
            optimizer.step()
            running_loss += loss.mean().item()

        print('Epoch {}, loss = {:.3f}'.format(epoch, running_loss/len(loader)))
    
    print('Training Finished')

model = PoseHRNet().cuda()
model = model.cuda(0)
if torch.cuda.device_count() > 1:
	  model = nn.DataParallel(model) 
criterion = nn.MSELoss(reduction='none')
optimizer = optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9)

## train
train(model=model, n_epoch=3, loader=train_dataloader, optimizer=optimizer, criterion=criterion, device="cuda")
torch.save(model, PATH)
## test
#resnet_acc = test(resnet_model, testloader, device="cuda")

#print('ResNet Test accuracy: {:.2f}%'.format(resnet_acc))
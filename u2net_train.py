import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import PatSegDataset, Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

import horovod.torch as hvd
# ------- 0. prepare distributed GPU computation --------
hvd.init()
torch.cuda.set_device(hvd.local_rank())

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'
anno_file = "./data/person_detection_and_tracking_results_drop-Gaitparams_PD.pkl"
frame_root = "/data/GaitData/RawFrames"
model_dir = f"/data/GaitData/checkpoints/{model_name}/"
pretrained_model = "saved_models/u2net_human_seg/u2net_human_seg.pth"

if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)
        
epoch_num = 100000
batch_size_train = 32
batch_size_val = 1
train_num = 0
val_num = 0

salobj_dataset = PatSegDataset(anno_file, frame_root, 
                   transform=transforms.Compose([
                       RescaleT(320),
                       RandomCrop(288),
                       ToTensorLab(flag=0)]))
sampler = torch.utils.data.distributed.DistributedSampler(
    salobj_dataset,
    num_replicas=hvd.size(),
    rank=hvd.rank(), shuffle=True)
salobj_dataloader = DataLoader(salobj_dataset, 
                               batch_size=batch_size_train,
                               sampler=sampler,
                               num_workers=8)
train_num = len(salobj_dataset)

# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(3, 1)
    # load human-segmentation pretrained model
    net.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
elif(model_name=='u2netp'):
    net = U2NETP(3,1)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=net.named_parameters())
hvd.broadcast_parameters(
    net.state_dict(),
    root_rank=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 1000 # save the model every 1000 iterations

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * (batch_size_train * hvd.size()), train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:

            torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0


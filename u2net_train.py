import os
import sys
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
import h5py

from data_loader import PatSegDataset, Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

import horovod.torch as hvd

from collections import deque

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
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss

class EarlyStop:
    def __init__(self, es_patience=5, verbose=True):
        self.es_patience = es_patience
        self.q = deque(maxlen=es_patience)
        self.count = 0
        self.verbose = verbose
    
    def step(self, val_loss):
        self.q.append(val_loss)
        if val_loss > min(self.q):
            self.count += 1
        if self.count > self.es_patience:
            if self.verbose:
                print("Early stoppping!!!!")
                print(f"[{val_loss}] es_count : ", self.count, self.q)
            return True
        
        return False

@torch.no_grad()
def evaluate(net, dataloader, save_features=False):
    ds = dataloader.dataset
    split_ = ds.split_

    print("Start evaluation!")
    net.eval()
    
    running_loss = 0.0
    running_tar_loss = 0.0
    import tqdm
    for i, data in enumerate(tqdm.tqdm(dataloader, desc=split_)):
        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        if save_features:
            # TODO: save encoder outputs
            feats = net(inputs_v)[-1] # last encoder output
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            idxs = data['imidx'].data.view(-1).cpu().numpy()
            feats_np = feats.data.cpu().numpy()
            with h5py.File(f'/data/GaitData/u2net_feats_{split_}.hdf5', 'a') as f:
                file_index = ds.anno.path.iloc[idxs].str.split('/').str[-2:].str.join('/').values
                feats_dset, index_dset = f.get('feats'), f.get('index')
                if feats_dset is None:
                    feats_dset = f.create_dataset('feats', shape=(len(ds), *feats_np.shape[1:]))
                if index_dset is None:
                    index_dset = f.create_dataset('index', shape=(len(ds),), dtype=h5py.special_dtype(vlen=str))
                feats_dset[idxs] = feats_np
                index_dset[idxs] = file_index
        else:
            # forward only
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss
    
    if save_features:
        return
        
    val_loss = running_loss / len(val_dataloader.dataset)
    val_tar_loss = running_tar_loss / len(val_dataloader.dataset)
    
    return val_loss, val_tar_loss

if __name__ == "__main__":
    # ------- 0. prepare distributed GPU computation --------
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # ------- 1. set the directory of training dataset --------

    model_name = 'u2net' #'u2netp'
    anno_file = "./data/person_detection_and_tracking_results_drop-Gaitparams_PD.pkl"
    target_file = "./data/targets_dataframe-Gaitparams_PD.pkl"
    frame_root = "/data/GaitData/frames"
    model_dir = f"/data/GaitData/checkpoints/{model_name}/"
    pretrained_model = "saved_models/u2net_human_seg/u2net_human_seg.pth"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
            
    epoch_num = 100000
    batch_size = 32
    train_num = 0
    val_num = 0

    train_dataset = PatSegDataset(anno_file, frame_root, target_file=target_file,
                    transform=transforms.Compose([
                        RescaleT(320),
                        RandomCrop(288),
                        ToTensorLab(flag=0)]), split_='train')
    val_dataset = PatSegDataset(anno_file, frame_root, target_file=target_file,
                    transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]), split_='val')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(), shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(), shuffle=False)

    train_dataloader = DataLoader(train_dataset, 
                                batch_size=batch_size,
                                sampler=train_sampler,
                                num_workers=8)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size,
                                sampler=val_sampler,
                                num_workers=8)

    train_num = len(train_dataset)

    # ------- 2. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 1)
        # load human-segmentation pretrained model
        net.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 3. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=net.named_parameters())
    hvd.broadcast_parameters(
        net.state_dict(),
        root_rank=0)

    # ------- 4. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 1000 # save model every 1000 iterations
    best_val_loss = 0.0
    es_patience = 5 # 5 times patience for early stopping
    es = EarlyStop(es_patience, verbose=True)

    for epoch in range(0, epoch_num):
        net.train()  # resume train

        for i, data in enumerate(train_dataloader):
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
            epoch + 1, epoch_num, (i + 1) * (batch_size * hvd.size()), train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:            
                val_loss, val_tar_loss = evaluate(net, val_dataloader)
                
                print("[epoch: %3d/%3d, ite: %d] val loss: %3f, tar: %3f " % (epoch + 1, epoch_num, ite_num, val_loss, val_tar_loss))

                
                # finish training when early stop is triggered
                if es.step(val_loss):
                    sys.exit(0)
                    
                if val_loss > best_val_loss:
                    # save best model
                    torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                    
                    # update best_val_loss
                    best_val_loss = val_loss
                    
                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0
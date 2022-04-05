import torch
from data_loader import PatSegDataset, RescaleT, ToTensorLab
from model.u2net import U2NET
from u2net_train import evaluate
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import h5py

ANNO_FILE = "./data/person_detection_and_tracking_results_drop-Gaitparams_PD.pkl"
FRAME_ROOT = "/data/GaitData/frames"
MODEL_DIR = "/data/GaitData/checkpoints/u2net/u2net_bce_itr_1000_train_1.462870_tar_0.205207.pth" # early stoped model (4 epoch, 1000 iter)

# load pretrained model
net = U2NET(3,1,mode='feature')
if torch.cuda.is_available():
    net.load_state_dict(torch.load(MODEL_DIR, map_location='cpu'))
    net.cuda()
net.eval() # eval mode

# load data and evaluate
batch_size = 100

test_transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
for split_ in ['val', 'test', 'train']:
    ds = PatSegDataset(ANNO_FILE, FRAME_ROOT, transform=test_transform, split_=split_)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=8, shuffle=False)
    evaluate(net, dl, save_features=True)
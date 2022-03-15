import numpy as np
import streamlit as st
import time, os
import torch
from torchvision import transforms
from torch.autograd import Variable
from data_loader import PatSegDataset, RescaleT, ToTensorLab
from model.u2net import U2NET
from u2net_human_seg_test import normPRED
from utils import get_detection, load_annotation
from PIL import Image
from skimage import io

ANNO_FILE = "./data/person_detection_and_tracking_results_drop-Gaitparams_PD.pkl"
FRAME_ROOT = "/data/GaitData/RawFrames"
# MODEL_DIR = "/data/GaitData/checkpoints/u2net/u2net_bce_itr_6000_train_0.168102_tar_0.022536.pth" # lightly pretrained model (2 epoch)
MODEL_DIR = "/data/GaitData/checkpoints/u2net/u2net_bce_itr_4000_train_0.184121_tar_0.024561.pth" # lightly pretrained model (1 epoch)

# MODEL_DIR = "/data/GaitData/checkpoints/u2net/u2net_bce_itr_33000_train_0.110568_tar_0.014451.pth" # heavily pretrained model (10 epoch)

# UI placeholders
slider_ph = st.empty()
filepath_ph = st.empty()
image_ph = st.empty()
pred_ph = st.empty()

@st.cache
def load_data():
    """
        Load dataset and model
    """
    test_dataset = PatSegDataset(ANNO_FILE, FRAME_ROOT, transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)]))
    
    net = U2NET(3,1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(MODEL_DIR, map_location='cuda:0'))
        net.cuda()
    net.eval()

    return test_dataset, net

def render_img(value):
    anno = test_dataset.anno  # annotation dataframe
    path = anno.path.iloc[value]
    # filepath_ph.text(path) # display filename
    filepath_ph.markdown(f'<p style="font-family:sans-serif; color:Lime; font-size: 24px;">{path}</p>', unsafe_allow_html=True)
    detection = get_detection(anno, path, visualize=True)
    detection = detection.resize((detection.width//2,detection.height//2),resample=Image.BILINEAR)
    image_ph.image(detection)
    
def render_result(value):
    sample = test_dataset[value]
    test_img = sample['image'].float()
    if torch.cuda.is_available():
        test_img = Variable(test_img.cuda())
    else:
        test_img = Variable(test_img)
        
    d1,d2,d3,d4,d5,d6,d7= net(test_img[None])

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    # TODO: otsu threshold
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(predict_np)
    binary = (predict_np > thresh).astype(float)
    
    im = Image.fromarray(binary*255).convert('RGB')
    img_name = test_dataset.anno.path.iloc[value] # original image
    image = io.imread(img_name) 
    imo = im.resize((image.shape[1]//2,image.shape[0]//2),resample=Image.BILINEAR)
    
    pred_ph.image(imo)
    del d1,d2,d3,d4,d5,d6,d7

data_load_state = st.text('Loading data...')
test_dataset, net = load_data()
data_load_state.text("Done! (using st.cache)")
data_load_state.text("")

# session-state
if 'idx' not in st.session_state:
    st.session_state['idx'] = 0

search_str = st.text_input('Search', '/data/GaitData/RawFrames/2074951_test_1_trial_3/thumb0174.jpg')
st.session_state['idx'] = test_dataset.anno.query("path == @search_str").index.values.item()

value = slider_ph.slider("Frame Index", 0, len(test_dataset), st.session_state['idx'], 1)

render_img(value)
render_result(value)

if st.button('PLAY'):
    for x in range(50):
        time.sleep(.3)
        value = slider_ph.slider("Frame Index", 0, len(test_dataset), value + 1, 1)
        render_img(value)
        # update session-state
        st.session_state['idx'] = value
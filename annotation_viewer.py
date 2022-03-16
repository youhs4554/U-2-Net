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
from skimage.filters import threshold_otsu

ANNO_FILE = "./data/person_detection_and_tracking_results_drop-Gaitparams_PD.pkl"
FRAME_ROOT = "/data/GaitData/RawFrames"
MODEL_DIR = "/data/GaitData/checkpoints/u2net/u2net_bce_itr_1000_train_1.462870_tar_0.205207.pth" # early stoped model (4 epoch, 1000 iter)

# UI placeholders
slider_ph = st.empty()
col1, col2 = st.columns(2)
with col1:
    image_ph = st.empty()
with col2:
    pred_ph = st.empty()
    
st.markdown(f'<p style="font-family:sans-serif; color:yellow; font-size: 24px;">File name</p>', unsafe_allow_html=True)
filepath_ph = st.empty()

@st.cache
def load_data():
    """
        Load dataset and model
    """
    test_dataset = PatSegDataset(ANNO_FILE, FRAME_ROOT, transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)]), split_='test')
    
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
    filepath_ph.markdown(f'<p style="font-family:sans-serif; font-size: 18px;">{path}</p>', unsafe_allow_html=True)
    detection = get_detection(anno, path, visualize=True)
    detection = detection.resize((detection.width,detection.height),resample=Image.BILINEAR)
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
    
    result = predict_np
    
    st.sidebar.text("Binarization (using Otsu)")
    col1, col2, *_ = st.sidebar.columns(5)
    with col1:
        if st.button('On', kwargs={'width': 50}):
            # TODO: control with btn
            thresh = threshold_otsu(predict_np)
            result = (predict_np > thresh).astype(float)
    with col2:
        if st.button('Off'):
            result = predict_np
    
    im = Image.fromarray(result*255).convert('RGB')
    img_name = test_dataset.anno.path.iloc[value] # original image
    image = io.imread(img_name) 
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    
    pred_ph.image(imo)
    del d1,d2,d3,d4,d5,d6,d7

data_load_state = st.text('Loading data...')
test_dataset, net = load_data()
data_load_state.text("Done! (using st.cache)")
data_load_state.text("")

# session-state
if 'idx' not in st.session_state:
    st.session_state['idx'] = 0

search_str = os.path.join(FRAME_ROOT, st.sidebar.text_input('Search', '2074951_test_1_trial_3/thumb0174.jpg'))
anno_copy = test_dataset.anno.copy()
anno_copy.reset_index(inplace=True)
matched = anno_copy.query("path == @search_str").index
if len(matched) > 0:
    st.session_state['idx'] = matched.values.item()
else:
    st.warning('{} is not found!'.format(search_str))
    st.session_state['idx'] = 0


value = slider_ph.slider("Index", 0, len(test_dataset), st.session_state['idx'], 1)

render_img(value)
render_result(value)

st.sidebar.text('Control')
play_btn = st.sidebar.button('Play')
if play_btn:
    for x in range(50):
        time.sleep(.3)
        value = slider_ph.slider("Index", 0, len(test_dataset), value + 1, 1)
        render_img(value)
        st.session_state['idx'] = value # update session-state
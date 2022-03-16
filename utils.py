import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def tensor2img(t):
    if t.size(0) == 1:
        # Binary mask label
        t = t.squeeze(0)
    if t.dtype == torch.double and t.min() < 0 and t.max() > 1:
        # RGB -> denormalize
        # t: normalized float img tensor. shape: (3,h,w) 
        mean_t = torch.tensor(mean).view(-1,1,1) # (3,1,1)
        std_t = torch.tensor(std).view(-1,1,1) # (3,1,1)
        t = t * std_t + mean_t
        t = t.permute(1,2,0) # (h,w,3)
        
    t *= 255
    np_img = t.detach().cpu().numpy().astype(np.uint8)
    return Image.fromarray(np_img)

def show(im, cmap=None):
    if isinstance(im, Image.Image):
        im = im
    if isinstance(im, str):
        im = Image.open(im)
    elif isinstance(im, torch.Tensor):
        im = tensor2img(im)
    plt.imshow(im, cmap=cmap)
        
def get_detection(anno, filename, visualize=False):
    """
        Detection box is overlayed
    """
    path, det = anno.query(f'path == @filename').iloc[0]
    
    im = Image.open(path)
    xmin, ymin, xmax, ymax = eval(det)

    draw = ImageDraw.Draw(im)
    draw.rectangle((xmin, ymin, xmax, ymax), outline='lime', width=4)
    if visualize:
        show(im)
    return im

def get_mask_label(anno, filename, visualize=False):
    """
        Binary coarse mask (i.e.,box)
    """
    path, det = anno.query(f'path == @filename').iloc[0]

    im = Image.open(path)
    xmin, ymin, xmax, ymax = eval(det)

    # mask image (target of u2net)
    mask = Image.new('L', (im.width,im.height))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((xmin, ymin, xmax, ymax), fill='white')
    if visualize:
        show(mask, cmap='gray')
    return mask

def load_annotation(anno_file, frame_root=None):
    """
        Load data and format annotation file
    """
    anno = pd.read_pickle(anno_file)
    anno.reset_index(drop=True, inplace=True)
    
    if frame_root is not None:
        format_fn = lambda row: f"{frame_root}/{row[0]}/thumb{int(row[1]):04d}.jpg"
        anno['path'] = anno.apply(format_fn, axis=1)
    else:
        anno['path'] = anno.vids

    del anno['vids']
    del anno['idx']

    anno = anno[['path', 'pos']]
    print("Total: ", len(anno))
    
    return anno

def split_by_video(anno, test_size=0.05, random_state=27):
    """
        Split by name of video (i.e., to reduce redundancy)
    """
    fn = lambda row: os.path.dirname(row)
    video_names = anno.path.apply(fn)
    
    anno_copy = anno.copy()
    anno_copy['video_name'] = video_names
    anno_copy.set_index('video_name', inplace=True)
    
    train_videos, test_videos = train_test_split(video_names.unique(), 
                                                 test_size=test_size, random_state=random_state)
    train_videos, val_videos = train_test_split(train_videos, 
                                                test_size=test_size, random_state=random_state)
    train_anno, val_anno, test_anno = anno_copy.loc[train_videos], anno_copy.loc[val_videos], anno_copy.loc[test_videos]
    
    # statistics
    print("[FRAMES] - train: {}, val: {}, test: {}".format(len(train_anno), len(val_anno), len(test_anno)))
    print("[VIDEOS] - train: {}, val: {}, test: {}".format(len(train_anno.index.unique()), len(val_anno.index.unique()), len(test_anno.index.unique())))
    
    return {
        'train': train_anno,
        'val': val_anno,
        'test': test_anno,
    }
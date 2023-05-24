# this "functions.py" is slightly different from the original one, it is adapted for the use of these scripts

from PIL import Image
import numpy as np
from os import listdir, makedirs
from os.path import join, isdir
import shutil
import random
import tkinter as tk
from tkVideoPlayer import TkinterVideo
from time import time

def create_directories(folder, verbose = True):
    """Will try to create the full path 'folder'. If already existing will do nothing"""
    try:
        makedirs(folder)
        if verbose: print(f'created folder {folder}')
    except:
        None

def remove_directory(folder, ask = True, notify = True):
    """Removes the folder 'folder' and all its contents, if such a folder exists. Use carefully.
    If 'ask' = True it will ask confirmation in the terminal before removing; if 'notify' = True will print the message "The folder "'+folder+'" was removed!". """
    if isdir(folder):
        if ask:
            input_ = str(input('\nYou want to delete the folder "'+folder+'" and all its contents? [y/N] '))
        if not ask or input_ == 'y' or input_ == 'Y':
            try:
                shutil.rmtree(folder)
                removed_ = True
            except:
                assert True, "Error: it was not possible to remove the folder. This message should not appear in any situation, check carefully what went wrong."
        else:
            removed_ = False
    else:
        removed_ = True # folder was not there
    
    if removed_ and notify:    print('\nThe folder "'+folder+'" was removed!\n')
    return removed_

print_step = lambda i, text: print(f"\n> STEP {i} - \t{text}")

################ 
### GENERATE ###
################

from perlin_noise import generate_perlin_noise_2d
from tqdm import tqdm
import numpy as np

def generate_img(folder, mask, parameters, i):
    """Subfunction of perlin_shapes.
    Creates artificially the data given the binary matrix 'mask' and saves it as image in 'folder' as 'i'.png.
    'parameters' should be a tuple containing (mu1, sigma1, mu2, sigma2), for example (50, 20, 100, 10).
    This function will generate two white noise matrices with the parameters (mu1, sigma1) and (mu2, sigma2), and will replace the 0's and the 1's in 'mask' with the respective white noises.
    It is required that mu1 <= mu2, to preserve the property that the masks (1's) are "whiter" then the background (0's)."""

    mu1, sigma1, mu2, sigma2 = parameters
    assert mu1<=mu2, "mu1 should not be bigger than mu2!"
    assert 0<=mu1<=255 and 0<=mu2<=255, "mu1 and mu2 should be between 0 and 255!"

    noise_False = np.clip(np.random.normal(mu1, sigma1, size = mask.shape),0,255) # clipping is necessary to avoid issues when converting to uint8: "img = img.astype(np.uint8)". The min(x,255) may be unnecessary
    noise_True  = np.clip(np.random.normal(mu2, sigma2, size = mask.shape),0,255) ## 15% elapsed time
    img = mask * noise_True + (1 - mask)*noise_False

    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    # im.save(folder + str(i) + '.png') ## 20% elapsed time
    
    ###
    rgbimg = Image.new("RGB", (64,64)) # set to 'L','RGB', ...
    rgbimg.paste(im)
    rgbimg.save(folder + str(i) + '.png')
    ###

def generate_label(folder, mask, i):
    """Subfunction of perlin_shapes.
    Just saves 'mask' in 'folder' as 'i'.png"""
    im = Image.fromarray(mask)
    im.save(folder + str(i) + '.png')

def perlin_shapes(folder,n_img, parameters = None, verbose = False):    
    """
    Example of use: perlin_shapes('tmp/',100, parameters = (85,50,170,50)).
    This function generates a randomized dataset for image segmentation. A perlin noise is generated, and all points whose absolute value is greater than a set threshold are assigned the value 1, otherwise 0.
    This way the masks matrices are generated. Then the masks are feeded into the functions generate_img and generate_label, (see the respective functions descriptions).
    'n_img' is the number of images generated. The images are created via Perlin noise.
    imgs are saved under subfolder 'img' and masks are saved under subfolder 'label'.
    """

    # create directories:
    create_directories(folder+'img', verbose = False)
    create_directories(folder+'label', verbose = False)
    
    for i in tqdm(range(n_img),disable = not verbose):
        
        # generate mask with perlin noise:
        prln = generate_perlin_noise_2d((64,64),(2,2)) ## 40% of elapsed time
        threshold = 0.4
        mask = (abs(prln) > threshold)
        
        # create and save image and/or save mask in folder:
        generate_img(folder+'img/', mask, parameters, i)
        generate_label(folder+'label/', mask, i)


import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from PIL import Image

def perlin_shapes_show_sample(n_imgs, parameters_list, mus_differences):    
    """
    Just a function to visualize the datasets.
    """
    n = len(parameters_list)
    n_rows = n
    n_cols = n_imgs

    rcParams['figure.figsize'] = n_cols, n_rows*1.2 # set image width and height
    fig, axs = plt.subplots(n_rows,n_cols)

    subfigs = fig.subfigures(nrows=n_rows, ncols=1)
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(f'$\mu_2 - \mu_1 = {mus_differences[i]}$')
        axs = subfig.subplots(nrows=1, ncols=n_cols)
        mu1,sigma1,mu2,sigma2 = parameters_list[i]
        for j, ax in enumerate(axs):
            prln = generate_perlin_noise_2d((64,64),(2,2))
            threshold = 0.4
            mask = (abs(prln) > threshold)

            noise_False = np.clip(np.random.normal(mu1, sigma1, size = mask.shape),0,255) # clipping is necessary to avoid issues when converting to uint8: "img = img.astype(np.uint8)". The min(x,255) may be unnecessary
            noise_True  = np.clip(np.random.normal(mu2, sigma2, size = mask.shape),0,255) ## 15% elapsed time
            img = mask * noise_True + (1 - mask)*noise_False
            img = img.astype(np.uint8)
            im = Image.fromarray(img)
            rgbimg = Image.new("RGB", (64,64)) # set to 'L','RGB', ...
            rgbimg.paste(im)

            ax.imshow(rgbimg)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]) # remove ticks

    plt.show('data_toydataset/samples.png')

    rcParams.update(rcParamsDefault) # reset parameters

############
### UNET ###
############

def IoU(mask1, mask2, value):
    im1 = Image.open(mask1)
    im2 = Image.open(mask2)
    matrix1 = np.array(im1) == value
    matrix2 = np.array(im2) == value
    intersection = np.sum(np.logical_and(matrix1,matrix2))
    union = np.sum(np.logical_or(matrix1,matrix2))
    if union == 0:
        return 0.0
    return intersection/union

def avg_IoU(folder1, folder2, value):
    IoU_list = []
    for file in listdir(folder1):
        IoU_list.append(IoU(join(folder1, file),join(folder2, file), value))
    return np.mean(IoU_list)

from unet.unet_model import UNet
import torch

def BN_adapt(model_root, dataset, device, saving_root):
    model = UNet(n_channels=3, n_classes=2).to(device=device)
    state_dict = torch.load(model_root, map_location=device)
    del state_dict['mask_values']
    model.load_state_dict(state_dict)

    model.train()
    with torch.no_grad():
        for batch in dataset :
            images = batch['image'].to(device=device, dtype=torch.float32)
            _ = model(images)
    torch.save(model.state_dict(), saving_root)

################
### GRAPH 3D ###
################

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def graph_3d(d, mus_differences, filename, title, show = False, is_diff = False):
    n = int(len(d)**0.5) ## somewhat unnecessary but meh...
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = np.arange(0,n,1.0)
    y = np.arange(0,n,1.0)
    Z = np.zeros((n,n))

    X,Y = np.meshgrid(x, y)

    for datapoint in d:
        Z[datapoint] = d[datapoint]
    Z = Z.T # Z needs to be transposed because of the way ax.plot_surface works

    # set title, axis labels, and axis ticks
    plt.title(title, fontsize = 20)
    ax.set_xlabel('Source $\mu_2 - \mu_1$', fontsize=12)
    ax.set_ylabel('Target $\mu_2 - \mu_1$', fontsize=12)
    ax.set_zlabel('IoU_difference' if is_diff else 'IoU', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(mus_differences)
    ax.set_yticks(y)
    ax.set_yticklabels(mus_differences)

    # set xyz ranges and point of view to keep visualization consistent
    ax.set_xlim(0, n-1)
    ax.set_ylim(0, n-1)
    if not is_diff:
        ax.set_zlim(0, 1)
    ax.view_init(elev=30., azim=225.) # (degrees)
    
    # add surface
    surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm, antialiased=False)

    # add dots on negative points if is_diff
    
    if is_diff:
        d_neg = {key:d[key] for key in d if d[key] < 0}
        x_,y_,z_ = [],[],[]
        for key in d_neg:
            x_.append(key[0])
            y_.append(key[1])
            z_.append(d_neg[key])

        # x, y = d_neg.keys()
        ax.scatter(x_,y_,z_, c='red')

    plt.savefig(f'data_toydataset/{filename}.png')
    if show == True:
        plt.show()


import plotly.graph_objects as go

def graph_3d_plotly(d, is_diff = False, mus_differences = None):

    n = int(len(d)**0.5)
    mus = bool(mus_differences)
    if not mus:
        mus_differences = [str(i) for i in range(n)]

    x = np.arange(0,n,1.0)
    y = np.arange(0,n,1.0)
    Z = np.zeros((n,n))
    
    X,Y = np.meshgrid(x, y)
    for datapoint in d:
        Z[datapoint] = d[datapoint]
    Z = Z.T # Z needs to be transposed because of the way ax.plot_surface works
    
    Z_zeros = np.zeros((n,n))

    d_neg = {key:d[key] for key in d if d[key] < 0}
    x_,y_,z_ = [],[],[]
    for key in d_neg:
        x_.append(key[0])
        y_.append(key[1])
        z_.append(d_neg[key])

    fig = go.Figure([
    go.Scatter3d(
    x = x_,
    y = y_,
    z = z_,
    mode = "markers",
    marker=dict(size=3,line=dict(width=0), color = "red"),
    ),
    go.Surface(
    x = x,
    y = y,
    z = Z_zeros,
    opacity=0.5,
    colorscale= "Blues"
    ),
    go.Surface(
    contours = {
        "x": {"show": True, "start": 0, "end": n, "size": 1, "color":"rgba(147,112,219,255)"},
        "y": {"show": True, "start": 0, "end": n, "size": 1, "color":"rgba(147,112,219,255)"},
    },
    x = x,
    y = y,
    z = Z),
    ]
    )
    
    fig.update_layout(
            scene = {
                "xaxis_title": "Source mu2 - mu1" if mus else "",
                "yaxis_title": "Target mu2 - mu1" if mus else "",
                "zaxis_title": "IoU" if not is_diff else "IoU_diff",
                "xaxis": {"range": [0,n-1], "tickvals": list(range(n+1)), "tickmode":"array", "ticktext" : mus_differences},
                "yaxis": {"range": [0,n-1], "tickvals": list(range(n+1)), "tickmode":"array", "ticktext" : mus_differences},
                "zaxis": {"range": [0,1]} if not is_diff else {},
                'camera_eye': {"x": -2, "y": -2, "z": 2},
                "aspectratio": {"x": 1, "y": 1, "z": 1}
            })
    fig.show()

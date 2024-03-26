import math
import numpy as np
import torch
import sys, os, glob

from skimage.color import rgb2lab
from lib.dataset.dataset import normalize_lab 
from lib.omni.omni import to_unit_vectors, hammersley_points_spherical_to_pixels, create_spherical_label_map
import lib.omni.enforce_connectivity
from lib.omni._enforce_connectivity import _enforce_connectivity_labels_360
from lib.ssn.ssn import sparse_ssn_iter

@torch.no_grad()
def inference(image, nspix, n_iter, fdim=None, color_scale=0.6, pos_scale=10, weight=None, enforce_connectivity=True):
    """
    generate superpixels

    Args:
        image: numpy.ndarray
            An array of shape (h, w, c)
        nspix: int
            number of superpixels
        n_iter: int
            number of iterations
        fdim (optional): int
            feature dimension for supervised setting
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        weight: state_dict
            pretrained weight
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing

    Return:
        labels: numpy.ndarray
            An array of shape (h, w)
    """

    height, width = image.shape[:2]
    batchsize = 1

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)    

    #Input features
    yv, xv = torch.meshgrid(torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)) # indexing='xy')
    pixel_points = torch.stack((xv.flatten(), yv.flatten()), dim=1)
    coords3D = to_unit_vectors(width, height, pixel_points).permute(1,0).to("cuda")
    coords3D = coords3D[None].repeat(batchsize, 1, 1, 1).float()

    image = rgb2lab(image).astype(np.float32)
    image = normalize_lab(image)
    image = torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda").float()

    inputs = torch.cat([color_scale*image, pos_scale*coords3D.reshape(batchsize, 3, height, width)], 1)

    #Initial label map
    num_spixels_width = int(math.sqrt(nspix * width / height))
    num_spixels_height = int(math.sqrt(nspix * height / width))
    num_points = num_spixels_width*num_spixels_height
    centers = hammersley_points_spherical_to_pixels(num_points, width, height)
    init_label_map, centers3D = create_spherical_label_map(centers, width, height)
    init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

    #Model loading
    if weight is not None:
        from model import SSNModel
        model = SSNModel(fdim, nspix, n_iter).to("cuda")
        model.load_state_dict(torch.load(weight))
        model.eval()
    else:
        model = lambda data, init_label_map, centers3D: sparse_ssn_iter(data, nspix, init_label_map, centers3D, n_iter)

    _, H, _ = model(inputs, init_label_map, centers3D)

    labels_preenforce = H.reshape(height, width).to("cpu").detach().numpy()

    if enforce_connectivity:
        segment_size = height * width / nspix
        min_size = int(0.15 * segment_size)
        max_size = int(10 * segment_size)
        labels = _enforce_connectivity_labels_360(labels_preenforce[None], min_size, max_size)[0]

    return labels


if __name__ == "__main__":

    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, type=str, help="/path/to/image")
    parser.add_argument("--folder", default=None, type=str, help="/path/to/folder containing images")
    parser.add_argument("--weight", default=None, type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=100, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.6, type=float)
    parser.add_argument("--pos_scale", default=10, type=float)
    args = parser.parse_args()

    if (args.image!=None):
        image = plt.imread(args.image)
        s = time.time()
        label = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)
        print(f"time {time.time() - s}sec")
        plt.imsave("res_" + str(args.pos_scale) + "_" + str(args.color_scale) + ".png", mark_boundaries(image, label))
        plt.imsave("res_" + str(args.pos_scale) + "_" + str(args.color_scale) + "_label.png", label)
        np.save("res_" + str(args.pos_scale) + "_" + str(args.color_scale) + "_label.npy", label)
    elif (args.folder!=None):
        s = time.time()
        ext = '*.jpg' 
        for images in glob.glob(os.path.join(args.folder, ext)): 
            print(images)
            image = plt.imread(images).astype('uint8')
            for nsp in np_range:
                label = inference(image, nsp, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)
                print(f"time {time.time() - s}sec")
                img_name = os.path.basename(os.path.normpath(images[0:-4]))
                plt.imsave("./res/"+img_name+'_'+str(nsp)+".png", mark_boundaries(image, label))
                res_dest = './res/'+img_name+'_'+str(nsp)+'_label.npy'
                np.save(res_dest, label)
    else:
        print("No image or folder given")

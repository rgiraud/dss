import os, math
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.meter import Meter
from model import SSNModel
from lib.dataset import dataset, augmentation
from lib.dataset.dataset import denormalize_lab
from lib.utils.loss import reconstruct_loss_with_cross_entropy, reconstruct_loss_with_mse
from lib.omni.omni import to_unit_vectors, hammersley_points_spherical_to_pixels, create_spherical_label_map

import lib.omni.enforce_connectivity
from lib.omni._enforce_connectivity import _enforce_connectivity_labels_360

from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

@torch.no_grad()
def eval(model, loader, init_label_map, centers3D, color_scale, pos_scale, device, nspix=500):
    def achievable_segmentation_accuracy(superpixel, label):
        """
        Function to calculate Achievable Segmentation Accuracy:
            ASA(S,G) = sum_j max_i |s_j \cap g_i| / sum_i |g_i|

        Args:
            input: superpixel image (H, W),
            output: ground-truth (H, W)
        """
        TP = 0
        unique_id = np.unique(superpixel)
        for uid in unique_id:
            mask = superpixel == uid
            label_hist = np.histogram(label[mask])
            maximum_regionsize = label_hist[0].max()
            TP += maximum_regionsize
        return TP / label.size


    model.eval()
    model.nspix = 200
    sum_asa = 0
    for data in loader:
        inputs, labels = data
        image_i = inputs.to("cpu").detach().numpy()

        inputs = inputs.to(device)
        labels = labels.to(device)

        batchsize, featsize, height, width = inputs.shape

        #Precompute 360 centers and initial label map
        num_spixels_width = int(math.sqrt(model.nspix * width / height))
        num_spixels_height = int(math.sqrt(model.nspix * height / width))
        num_points = num_spixels_width*num_spixels_height
        centers = hammersley_points_spherical_to_pixels(num_points, width, height)
        init_label_map, centers3D = create_spherical_label_map(centers, width, height)

        nspix_per_axis = int(math.sqrt(model.nspix))
        pos_scale_train = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)

        #Input features
        yv, xv = torch.meshgrid(torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)) # indexing='xy')
        pixel_points = torch.stack((xv.flatten(), yv.flatten()), dim=1)
        coords3D = to_unit_vectors(width, height, pixel_points).permute(1,0).to("cuda")
        coords3D = coords3D[None].repeat(batchsize, 1, 1, 1).float()

        inputs_rgb_coord = torch.cat([color_scale*inputs, pos_scale_train*coords3D.reshape(batchsize, 3, height, width)], 1)

        _, H, _ = model(inputs_rgb_coord, init_label_map, centers3D)

        H = H.reshape(batchsize, height, width)

        labels = labels.argmax(1).reshape(batchsize, height, width)

        asa = achievable_segmentation_accuracy(H.to("cpu").detach().numpy(), labels.to("cpu").numpy())
        sum_asa += asa

    model.train()
    return sum_asa / len(loader)


def update_param(data, model, init_label_map, centers3D, optimizer, compactness, color_scale, pos_scale, device, display_train=1):
    inputs, labels = data

    image = np.copy(inputs)
    inputs = inputs.to(device)
    labels = labels.to(device)

    batchsize, featsize, height, width = inputs.shape

    nspix_per_axis = int(math.sqrt(model.nspix))
    pos_scale_train = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)

    #Input features
    yv, xv = torch.meshgrid(torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)) # indexing='xy')
    pixel_points = torch.stack((xv.flatten(), yv.flatten()), dim=1)
    coords3D = to_unit_vectors(width, height, pixel_points).permute(1,0).to("cuda")
    coords3D = coords3D[None].repeat(batchsize, 1, 1).float()

    inputs = torch.cat([color_scale*inputs, pos_scale_train*coords3D.reshape(batchsize, 3, height, width)], 1)

    Q, H, feat = model(inputs, init_label_map, centers3D)

    #Loss
    recons_loss = reconstruct_loss_with_cross_entropy(Q, labels)
    compact_loss = reconstruct_loss_with_mse(Q, coords3D, H)
    loss = recons_loss + compactness * compact_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "reconstruction": recons_loss.item(), "compact": compact_loss.item()}


def train(cfg):

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    augment_color = augmentation.Compose([augmentation.RandomGaussianBlur(), augmentation.RandomGaussianNoise()])
    augment = augmentation.Compose([augmentation.RandomHorizontalFlip(), augmentation.RandomHorizontalShift(),
                                    augmentation.RandomCropandMirror(), augmentation.RandomPanoStretch()])


    train_dataset = dataset.load_dataset(cfg.root, geo_transforms=augment, color_transforms=augment_color)
    test_dataset = dataset.load_dataset(cfg.root, split="val")

    train_loader = DataLoader(train_dataset, cfg.batchsize, shuffle=True, drop_last=True, num_workers=cfg.nworkers)
    test_loader = DataLoader(test_dataset, 1, shuffle=False, drop_last=False)

    meter = Meter()

    inputs, _ = next(iter(train_loader))
    _, _, height, width = inputs.shape

    #Precompute 360 centers and initial label map
    num_spixels_width = int(math.sqrt(cfg.nspix * width / height))
    num_spixels_height = int(math.sqrt(cfg.nspix * height / width))
    num_points = num_spixels_width*num_spixels_height
    centers = hammersley_points_spherical_to_pixels(num_points, width, height)
    init_label_map, centers3D = create_spherical_label_map(centers, width, height)

    model = SSNModel(cfg.fdim, cfg.nspix, cfg.niter).to(device)
    optimizer = optim.Adam(model.parameters(), cfg.lr)
    iterations = 0

    max_val_asa = 0
    while iterations < cfg.train_iter:
        for data in train_loader:
            iterations += 1
            metric = update_param(data, model, init_label_map, centers3D, optimizer, cfg.compactness, cfg.color_scale, cfg.pos_scale,  device, display_train=display_train)
            meter.add(metric)
            state = meter.state(f"[{iterations}/{cfg.train_iter}]")
            print(state)
            if (iterations % cfg.test_interval) == 0:
                asa = eval(model, test_loader, init_label_map, centers3D, cfg.color_scale, cfg.pos_scale, device, cfg.nspix)
                print(f"validation asa {asa}")
                if asa > max_val_asa:
                    max_val_asa = asa
                    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "best_model_"+str(iterations)+".pth"))
            if iterations == cfg.train_iter:
                break

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_"+str(iterations)+".pth"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default="./data/BSR/BSDS500/data", type=str, help="/path/to/data")
    parser.add_argument("--out_dir", default="./log", type=str, help="/path/to/output directory")
    parser.add_argument("--batchsize", default=6, type=int)
    parser.add_argument("--nworkers", default=4, type=int, help="number of threads for CPU parallel")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--train_iter", default=500000, type=int)
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=5, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=200, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.6, type=float)
    parser.add_argument("--pos_scale", default=10, type=float)
    parser.add_argument("--compactness", default=1e0, type=float)
    parser.add_argument("--test_interval", default=1000, type=int)
    parser.add_argument("--weights", default=None, type=str, help="pretrained weights to load")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train(args)

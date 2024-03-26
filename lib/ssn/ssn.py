import math
import torch
import numpy as np

from .pair_wise_distance import PairwiseDistFunction
from ..utils.sparse_utils import naive_sparse_bmm
from ..omni.omni import hammersley_points_spherical_to_pixels, create_spherical_label_map


def mean_pooling_by_labels(features, label_map):
    batch_size, num_channels, height, width = features.size()
    num_labels = label_map.max() + 1

    features_flat = features.view(batch_size, num_channels, height * width)  
    label_map_flat = label_map.view(batch_size, height * width).to(features.device)  

    features_sum = torch.zeros(batch_size, num_channels, num_labels, device=features.device, dtype=features.dtype)
    features_count = torch.zeros(batch_size, num_labels, device=features.device, dtype=torch.float)

    for b in range(batch_size):
        indices = label_map_flat[b] + num_labels * torch.arange(num_channels, device=features.device).unsqueeze(-1)
        
        indices_flat = indices.view(-1)
        features_flat_b = features_flat[b].view(-1)

        features_sum_b = features_sum[b].view(num_channels * num_labels)
        features_sum_b.scatter_add_(0, indices_flat, features_flat_b)

        features_count_b = features_count[b]
        features_count_b.scatter_add_(0, label_map_flat[b], torch.ones_like(label_map_flat[b], dtype=torch.float))

    features_mean = features_sum / features_count.unsqueeze(1).clamp(min=1)  

    return features_mean.view(batch_size, num_channels, num_labels)




def calc_init_centroid(images, init_label_map):
    batchsize, channels, _, _ = images.shape

    centroids = mean_pooling_by_labels(images, init_label_map)
    init_label_map = init_label_map.reshape(batchsize, -1).type_as(centroids)
    centroids = centroids.reshape(batchsize, channels, -1)

    return centroids, init_label_map


@torch.no_grad()
def get_abs_indices(init_label_map, centers_3D, k=9, device="cuda"):
    b, n_pixel = init_label_map.shape

    dot_product = torch.mm(centers_3D.to(device), centers_3D.to(device).t()).to(device)
    _, relative_sp_indices = torch.topk(-dot_product, k=k, largest=False, dim=1)

    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = relative_sp_indices[init_label_map.long()].permute(0,2,1)

    abs_spix_indices = abs_spix_indices.reshape(-1).long()
    
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, abs_spix_indices):
    relative_label = affinity_matrix.max(1)[1]
    batch_size, num_pixels = relative_label.shape
    abs_spix_indices = abs_spix_indices.reshape(batch_size, 9, num_pixels)

    batch_indices = torch.arange(batch_size)[:, None].expand(-1, num_pixels)
    pixel_indices = torch.arange(num_pixels)[None, :].expand(batch_size, -1)
    
    label = abs_spix_indices[batch_indices, relative_label, pixel_indices]

    return label.long()


@torch.no_grad()
def sparse_ssn_iter(pixel_features, num_spixels, n_iter, init_label_map, centers3D):
    """
    computing assignment iterations with sparse matrix
    detailed process is in Algorithm 1, line 2 - 6
    NOTE: this function does NOT guarantee the backward computation.

    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        num_spixels: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """
    return ssn_iter(pixel_features, num_spixels, n_iter, init_label_map, centers3D, sparse=True)


def ssn_iter(pixel_features, num_spixels, n_iter, init_label_map, centers3D, sparse=False):
    """
    computing assignment iterations
    detailed process is in Algorithm 1, line 2 - 6

    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        num_spixels: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """
    batchsize, channels, height, width = pixel_features.shape
    num_spixels_width = int(math.sqrt(num_spixels * width / height))
    num_spixels_height = int(math.sqrt(num_spixels * height / width))

    init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)
    spixel_features, init_label_map = calc_init_centroid(pixel_features, init_label_map)
    abs_indices = get_abs_indices(init_label_map, centers3D)
    abs_spix_indices = abs_indices[1]
    
    #B x F x H x W
    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1) 
    permuted_pixel_features = pixel_features.permute(0, 2, 1)

    if not sparse:
        permuted_pixel_features = permuted_pixel_features.contiguous()

    for _ in range(n_iter):

        dist_matrix = PairwiseDistFunction.apply(pixel_features, spixel_features, abs_spix_indices.float(), num_spixels_width, num_spixels_height)

        affinity_matrix = (-dist_matrix).softmax(1)
        reshaped_affinity_matrix = affinity_matrix.reshape(-1)

        mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels) 
        sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask])

        if sparse:
            spixel_features = naive_sparse_bmm(sparse_abs_affinity, permuted_pixel_features) \
                / (torch.sparse.sum(sparse_abs_affinity, 2).to_dense()[..., None] + 1e-16)
            abs_affinity = sparse_abs_affinity
        else:
            abs_affinity = sparse_abs_affinity.to_dense().contiguous()
            spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) \
                / (abs_affinity.sum(2, keepdim=True) + 1e-16)

        spixel_features = spixel_features.permute(0, 2, 1)
        if not sparse:
            spixel_features = spixel_features.contiguous()

    hard_labels = get_hard_abs_labels(affinity_matrix, abs_spix_indices)

    return abs_affinity, hard_labels, spixel_features

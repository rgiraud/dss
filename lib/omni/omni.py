import torch
import math
import matplotlib.pyplot as plt


def to_unit_vectors(width, height, points):

    theta = (points[..., 0] / width) * 2 * math.pi
    phi = (points[..., 1] / height) * math.pi - math.pi / 2

    x = torch.cos(phi) * torch.cos(theta)
    y = torch.cos(phi) * torch.sin(theta)
    z = torch.sin(phi)
    return torch.stack((x, y, z), dim=-1)


def hammersley_points_spherical_to_pixels(num_points, width, height):

    #Generates Hammersley sampling
    x = torch.arange(1, num_points + 1) / num_points
    y = van_der_corput(num_points, 2)
    longitudes = 2 * math.pi * x - math.pi
    latitudes = torch.acos(2 * y - 1) - math.pi / 2
    
    #Normalization
    x_norm = (longitudes + math.pi) / (2 * math.pi)
    y_norm = (latitudes + math.pi / 2) / math.pi
    
    x_pixel = torch.floor(x_norm * width).to(torch.int32)
    y_pixel = torch.floor(y_norm * height).to(torch.int32)
    
    return torch.stack((x_pixel, y_pixel), dim=1)


def van_der_corput(n, base):
    sequence = torch.zeros(n)
    for i in range(n):
        j, k, s = i + 1, 0, 0.0
        while j > 0:
            s += (j % base) / (base ** (k + 1))
            j //= base
            k += 1
        sequence[i] = s
    
    return sequence


def create_spherical_label_map(hammersley_points, width, height, plot_map=0):

    centers_3D = to_unit_vectors(width, height, hammersley_points)

    yv, xv = torch.meshgrid(torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)) 
    pixel_points = torch.stack((xv.flatten(), yv.flatten()), dim=1) 
    pixel_vectors = to_unit_vectors(width, height, pixel_points)

    similarity = torch.mm(pixel_vectors, centers_3D.T)

    #Finds nearest cluster 
    labels = torch.argmax(similarity, dim=1).reshape(height, width)

    return labels, centers_3D



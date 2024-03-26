
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 256

#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__ pixel_features,
    const scalar_t* __restrict__ spixel_features,
    const scalar_t* __restrict__ abs_spix_indices, 
    scalar_t* __restrict__ dist_matrix,
    int batchsize, int channels, int num_pixels, int num_spixels,
    int num_spixels_w, int num_spixels_h
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batchsize * num_pixels * 9) return; 

    int cp = channels * num_pixels;
    int cs = channels * num_spixels;

    int b = index / (num_pixels * 9);
    int pixel_and_neighbor = index % (num_pixels * 9);
    int p = pixel_and_neighbor / 9;
    int spixel_offset = pixel_and_neighbor % 9; 

    int query_spixel_index = abs_spix_indices[b * (9 * num_pixels) + spixel_offset * num_pixels + p]; 

    if ((query_spixel_index<0) || (query_spixel_index>=num_spixels)) {
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = 1e16;
    }
    else {
        scalar_t sum_squared_diff = 0;
        for (int c = 0; c < channels; c++) {
            sum_squared_diff += pow(pixel_features[b * cp + c * num_pixels + p] -
                                    spixel_features[b * cs + c * num_spixels + query_spixel_index], 2);
        }
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = sum_squared_diff; 
    }
}

torch::Tensor forward_cuda(
    const torch::Tensor pixel_features,
    const torch::Tensor spixel_features,
    const torch::Tensor abs_spix_indices,
    torch::Tensor dist_matrix,
    int num_spixels_w, int num_spixels_h
){
    int batchsize = pixel_features.size(0);
    int channels = pixel_features.size(1);
    int num_pixels = pixel_features.size(2);
    int num_spixels = spixel_features.size(2);

    dim3 block((batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

    AT_DISPATCH_FLOATING_TYPES(dist_matrix.type(), "forward_kernel", ([&] {
        forward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            pixel_features.data<scalar_t>(),
            spixel_features.data<scalar_t>(),
            abs_spix_indices.data<scalar_t>(),
            dist_matrix.data<scalar_t>(),
            batchsize, channels, num_pixels,
            num_spixels, num_spixels_w, num_spixels_h
        );
    }));

    return dist_matrix;
}

template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* __restrict__ dist_matrix_grad,
    const scalar_t* __restrict__ pixel_features,
    const scalar_t* __restrict__ spixel_features,
    const scalar_t* __restrict__ abs_spix_indices, 
    scalar_t* __restrict__ pixel_feature_grad,
    scalar_t* __restrict__ spixel_feature_grad,
    int batchsize, int channels, int num_pixels, int num_spixels
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batchsize * num_pixels * 9) return;

    int cp = channels * num_pixels;
    int cs = channels * num_spixels;

    int b = index / (num_pixels * 9);
    int pixel_and_neighbor = index % (num_pixels * 9);
    int p = pixel_and_neighbor / 9; 
    int spixel_offset = pixel_and_neighbor % 9; 

    int query_spixel_index = abs_spix_indices[b * (9 * num_pixels) + spixel_offset * num_pixels + p];

    if ((query_spixel_index<0) || (query_spixel_index>=num_spixels))
        return;

    scalar_t dist_matrix_grad_val = dist_matrix_grad[b * (9 * num_pixels) + spixel_offset * num_pixels + p];

    for (int c = 0; c < channels; c++) {
        scalar_t pix_value = pixel_features[b * cp + c * num_pixels + p];
        scalar_t spix_value = spixel_features[b * cs + c * num_spixels + query_spixel_index];
        scalar_t diff = (pix_value - spix_value) * dist_matrix_grad_val;

        atomicAdd(&pixel_feature_grad[b * cp + c * num_pixels + p], 2 * diff);
        atomicAdd(&spixel_feature_grad[b * cs + c * num_spixels + query_spixel_index], -2 * diff);
    }
}


std::vector<torch::Tensor> backward_cuda(
    const torch::Tensor dist_matrix_grad,
    const torch::Tensor pixel_features,
    const torch::Tensor spixel_features,
    const torch::Tensor spixel_indices,
    torch::Tensor pixel_features_grad,
    torch::Tensor spixel_features_grad,
    int num_spixels_w, int num_spixels_h
){
    int batchsize = pixel_features.size(0);
    int channels = pixel_features.size(1);
    int num_pixels = pixel_features.size(2);
    int num_spixels = spixel_features.size(2);


    dim3 block((batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

    AT_DISPATCH_FLOATING_TYPES(pixel_features_grad.type(), "backward_kernel", ([&] {
        backward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            dist_matrix_grad.data<scalar_t>(),
            pixel_features.data<scalar_t>(),
            spixel_features.data<scalar_t>(),
            spixel_indices.data<scalar_t>(),
            pixel_features_grad.data<scalar_t>(),
            spixel_features_grad.data<scalar_t>(),
            batchsize, channels, num_pixels, num_spixels
        );
    }));

    return {pixel_features_grad, spixel_features_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_cuda, "pair_wise_distance forward");
  m.def("backward", &backward_cuda, "pair_wise_distance backward");
}

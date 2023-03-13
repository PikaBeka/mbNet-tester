#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/param.h>
#include <iostream>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include "cublas_utils.h"

#include "mbnet.h"

#define CHECK_CUDNN(expression)                                    \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS)                        \
        {                                                          \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

/*Function to fill input and weight matrix with random values*/
void fillWithValues(float *input, float *weight)
{
    srand(time(0));

    for (int b = 0; b < BATCH; b++)
    {
        for (int i = 0; i < C; i++)
        {
            for (int j = 0; j < HW; j++)
            {
                for (int k = 0; k < HW; k++)
                {
                    input[b * C * HW * HW + i * HW * HW + j * HW + k] = (float)(rand() % 100) - 100.0;
                    // input[i * HW * HW + j * HW + k] = 1.0f;
                }
            }
        }
    }

    for (int i = 0; i < K; i++)
    {
        for (int t = 0; t < C; t++)
        {
            for (int j = 0; j < RS; j++)
            {
                for (int k = 0; k < RS; k++)
                {
                    weight[i * (C * RS * RS) + t * (RS * RS) + j * RS + k] = (float)(rand() % 100) - 100.0;
                    // weight[i * (C * RS * RS) + t * (RS * RS) + j * RS + k] = 1.0f;
                }
            }
        }
    }
}

/*Function to verify the */
void verification(float *input, float *weight, float *output)
{
    for (int b = 0; b < BATCH; b++)
    {
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < PQ; j++)
            {
                for (int k = 0; k < PQ; k++)
                {
                    float tempC = 0.0f;
                    for (int l = 0; l < C; l++)
                    {
                        for (int m = 0; m < RS; m++)
                        {
                            for (int t = 0; t < RS; t++)
                            {
                                tempC += weight[i * C * RS * RS + l * RS * RS + m * RS + t] * input[b * C * HW * HW * l * HW * HW + (j + m) * HW + (k + t)];
                            }
                        }
                    }
                    if (abs(int(round(output[i * PQ * PQ + j * PQ + k]) - tempC)) > 3)
                    {
                        printf("The error is here. The actual result is %f, we get %f on (%d, %d, %d), the diff is %d\n", tempC, output[i * PQ * PQ + j * PQ + k], i, j, k, abs(int(round(output[i * PQ * PQ + j * PQ + k]) - tempC)));
                        exit(-1);
                    }
                }
            }
        }
    }
#if ARRAY_NAIVE
    printf("Array Naive convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", N, C, HW, HW, K, RS, RS, K, PQ, PQ);

#elif ARRAY_TILING
    printf("Array Tiling convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", N, C, HW, HW, K, RS, RS, K, PQ, PQ);

#elif DIRECT
#if CONV_SHARED
    printf("Shared direct convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", N, C, HW, HW, K, RS, RS, K, PQ, PQ);
#else
    printf("Global direct convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", N, C, HW, HW, K, RS, RS, K, PQ, PQ);
#endif

#elif CUDNN
    printf("CUDNN convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", N, C, HW, HW, K, RS, RS, K, PQ, PQ);
#else
#if GEMM_GLOBAL
    printf("Unroll globall gemm convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", N, C, HW, HW, K, RS, RS, K, PQ, PQ);
#else
    printf("Unroll cublass convolution finished. It is checked with %d images, and correct with image sizes (%d, %d, %d) and kernel (%d, %d, %d) resulting in (%d, %d, %d)\n", N, C, HW, HW, K, RS, RS, K, PQ, PQ);
#endif
#endif
}

/*-------------------------------------------------Array Naive-------------------------------------------------------------------*/
__global__ void convolution_naive(float input[C][HW][HW], float weight[K][C][RS][RS], float output[K][PQ][PQ])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int out_ch = idx % K;

    int output_x = (idx / K) % PQ;
    int output_y = (idx / K / PQ) % PQ;

    float tempC = 0.0f;

    for (int k = 0; k < C; k++)
    {
        for (int i = 0; i < RS; i++)
        {
            for (int j = 0; j < RS; j++)
            {
                tempC += weight[out_ch][k][i][j] * input[k][output_x + i][output_y + j];
            }
        }
    }

    output[out_ch][output_x][output_y] = tempC;
}

/*-------------------------------------------------Array Tiling-------------------------------------------------------------------*/
__global__ void convolution_tiling(float input[C][HW][HW], float weight[K][C][RS][RS], float output[K][PQ][PQ])
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    __shared__ float shm[C][TILE_S][TILE_S];

    int control = (row + LIM * blockIdx.y) * HW + col + LIM * blockIdx.x;

    if (control < HW * HW)
    {
        for (int z = 0; z < C; z++)
        {
            shm[z][row][col] = input[z][row + LIM * blockIdx.y][col + LIM * blockIdx.x];
        }
    }

    __syncthreads();

    float temp = 0.0f;

    if (row < LIM && col < LIM && col + LIM * blockIdx.x < PQ && row + blockIdx.y * LIM < PQ)
    {
        for (int k = 0; k < C; k++)
        {
            for (int i = 0; i < RS; i++)
            {
                for (int j = 0; j < RS; j++)
                {
                    temp += shm[k][row + i][col + j] * weight[blockIdx.z][k][i][j];
                }
            }
        }
        output[blockIdx.z][row + blockIdx.y * LIM][col + LIM * blockIdx.x] = temp;
    }
}

/*-------------------------------------------------Direct convolution-------------------------------------------------------------*/
__global__ void kernel_conv_filter(float input[C][HW][HW],
                                   float pre_output[K][PQ][PQ],
                                   float weight[K][C][RS][RS])
{
#if CONV_SHARED
    int tidx = threadIdx.x;
    int bIdx = blockIdx.x;

    __shared__ float sh_img[C][TILE_S][TILE_S];

    int img_row = tidx / TILE_S;
    int img_col = tidx % TILE_S;

    int bIdx_r = bIdx / GRID;
    int bIdx_c = bIdx % GRID;

    /* input image copy to shared memory */
    if (tidx < TILE_S * TILE_S)
    {
        for (int img_z = 0; img_z < C; img_z++)
            sh_img[img_z][img_row][img_col] = input[img_z][bIdx_r * LIM + img_row][bIdx_c * LIM + img_col];
    }

    __syncthreads();

    int ch = tidx / (LIM * LIM);
    int w_row = (tidx % (LIM * LIM)) / LIM;
    int w_col = (tidx % (LIM * LIM)) % LIM;

    float sum = 0;
    if (w_row < LIM && w_col < LIM && ch < K && bIdx_r * LIM + w_row < PQ && bIdx_c * LIM + w_col < PQ)
    {
        for (int k = 0; k < C; k++)
        {
            for (int i = 0; i < RS; i++)
                for (int j = 0; j < RS; j++)
                    sum += sh_img[k][w_row + i][w_col + j] * weight[ch][k][i][j];
        }
        pre_output[ch][bIdx_r * LIM + w_row][bIdx_c * LIM + w_col] = sum;
    }

#else
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int channel = idx % K;
    int output_x = (idx / K) % PQ;
    int output_y = (idx / K / PQ) % PQ;

    float tempC = 0.0f;
    for (int k = 0; k < C; k++)
    {
        for (int i = 0; i < RS; i++)
        {
            for (int j = 0; j < RS; j++)
            {
                tempC += weight[channel][k][i][j] * input[k][i + output_x][j + output_y];
            }
        }
    }
    if (idx < K * PQ * PQ)
        pre_output[channel][output_x][output_y] = tempC;
#endif
}

/*-------------------------------------------------Unrolling -----------------------------------------------------------------------*/
//*/
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
__global__ void im2col_gpu_kernel(const int n, const float *data_im,
                                  const int height, const int width, const int ksize,
                                  const int pad,
                                  const int stride,
                                  const int height_col, const int width_col,
                                  float *data_col)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < n; index += blockDim.x * gridDim.x)
    {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float *data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float *data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i)
        {
            for (int j = 0; j < ksize; ++j)
            {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void verify_im2col(float *A, float val)
{
    float maxError = 0.0f;

    int cnt = 0;
    for (int i = 0; i < RS * RS * PQ * PQ * C; i++)
    {
        maxError = max(abs(A[i] - val), maxError);
        if (maxError != 0)
            cnt++;
    }
    printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, RS * RS * PQ * PQ * C);
}

__global__ void ker2row_kernel(float weight_col[K][C * RS * RS], float weight[K][C][RS][RS])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int channel = idx % K;
    int z = (idx / K) % C;
    int x = (idx / K / C) % RS;
    int y = (idx / K / C / RS) % RS;

    if (idx < K * C * RS * RS)
    {
        weight_col[channel][z * RS * RS + x * RS + y] = weight[channel][z][x][y];
    }
}

void verify_ker2row(float *A, float val)
{
    float maxError = 0.0f;

    int cnt = 0;
    for (int i = 0; i < K * C * RS * RS; i++)
    {
        maxError = max(abs(A[i] - val), maxError);
        if (maxError != 0)
            cnt++;
    }
    printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, K * C * RS * RS);
}

__global__ void gemm_global_kernel(float matB[K][C * RS * RS], float matA[C * RS * RS][PQ * PQ], float matC[K][PQ * PQ])
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int x = idx % K;
    int y = (idx / K) % (PQ * PQ);

    float tempC = 0.0;
    if (idx < K * PQ * PQ)
    {
        for (int i = 0; i < C * RS * RS; i++)
        {
            tempC += matB[x][i] * matA[i][y];
        }
        matC[x][y] = tempC;
    }
}

void pass(float *input, float *weight, float *output)
{
    fillWithValues(input, weight);
    float *d_input, *d_weight, *d_output;

    cudaMalloc((void **)&d_input, BATCH * C * HW * HW * sizeof(float));
    cudaMalloc((void **)&d_weight, RS * RS * K * C * sizeof(float));
    cudaMalloc((void **)&d_output, BATCH * PQ * PQ * K * sizeof(float));

    cudaMemcpy(d_input, input, BATCH * C * HW * HW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, RS * RS * K * C * sizeof(float), cudaMemcpyHostToDevice);
    for (int batch = 0; batch < N / BATCH; batch++)
    {
#if ARRAY_NAIVE
        int threads = min(64, HW * HW);
        int total = K * (PQ * PQ);
        convolution_naive<<<(total + threads - 1) / threads, threads>>>((float(*)[HW][HW])d_input, (float(*)[C][RS][RS])d_weight, (float(*)[PQ][PQ])d_output);

#elif ARRAY_TILING
        dim3 threads(TILE_S, TILE_S);
        dim3 blocks((PQ + LIM - 1) / LIM, (PQ + LIM - 1) / LIM, K);
        convolution_tiling<<<blocks, threads>>>((float(*)[HW][HW])d_input, (float(*)[C][RS][RS])d_weight, (float(*)[PQ][PQ])d_output);

#elif DIRECT
#if CONV_SHARED
        const dim3 numBlocks(CONV_NB);
        const dim3 threadsPerBlock(CONV_TPB);
        kernel_conv_filter<<<numBlocks, threadsPerBlock>>>((float(*)[HW][HW])d_input,
#else
        int total = K * PQ * PQ;
        int threads = 64;
        kernel_conv_filter<<<(total + threads - 1) / threads, threads>>>((float(*)[HW][HW])d_input,
#endif
                                                           (float(*)[PQ][PQ])d_output,
                                                           (float(*)[C][RS][RS])d_weight);

#elif CUDNN
        cudnnHandle_t cudnn;
        CHECK_CUDNN(cudnnCreate(&cudnn));

        // Initialize CUDA
        cudaSetDevice(0);

        // Create input tensor
        cudnnTensorDescriptor_t input_descriptor;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               BATCH,
                                               C,
                                               HW,
                                               HW));

        // Create convolutional layer
        cudnnConvolutionDescriptor_t convolution_descriptor;
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                        0,
                                        0,
                                        1,
                                        1,
                                        1,
                                        1,
                                        CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT);

        // Create filter tensor
        cudnnFilterDescriptor_t filter_descriptor;
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
                                               CUDNN_DATA_FLOAT,
                                               CUDNN_TENSOR_NCHW,
                                               K,
                                               C,
                                               RS,
                                               RS));

        // Create output tensor
        int batch_size, channels, height, width;
        CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                          input_descriptor,
                                                          filter_descriptor,
                                                          &batch_size,
                                                          &channels,
                                                          &height,
                                                          &width));

        cudnnTensorDescriptor_t output_descriptor;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               batch_size,
                                               channels,
                                               height,
                                               width));

        // Allocate memory for workspace
        size_t workspace_size;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                            input_descriptor,
                                                            filter_descriptor,
                                                            convolution_descriptor,
                                                            output_descriptor,
                                                            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                                                            &workspace_size));
        void *workspace_data;
        cudaMalloc(&workspace_data, workspace_size);

        // Perform convolution
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUDNN(cudnnConvolutionForward(cudnn,
                                            &alpha,
                                            input_descriptor,
                                            d_input,
                                            filter_descriptor,
                                            d_weight,
                                            convolution_descriptor,
                                            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                                            workspace_data,
                                            workspace_size,
                                            &beta,
                                            output_descriptor,
                                            d_output));

        cudaFree(workspace_data);
        cudnnDestroyTensorDescriptor(input_descriptor);
        cudnnDestroyTensorDescriptor(output_descriptor);
        cudnnDestroyFilterDescriptor(filter_descriptor);
        cudnnDestroyConvolutionDescriptor(convolution_descriptor);
        cudnnDestroy(cudnn);
#else
        // im2col_gpu_kernel_ext<<<(N1+K1-1)/K1, K1>>>(PQ*PQ, d_input, HW, HW, RS, RS, 0, 0, STRIDE, STRIDE, 1, 1, PQ, PQ,ic_workspace);
        ///*
        float *im2col_A, *gemm_B, *gemm_C;

        cudaMalloc(&im2col_A, sizeof(float) * RS * RS * C * PQ * PQ);
        cudaMalloc(&gemm_B, sizeof(float) * K * C * RS * RS);
        cudaMalloc(&gemm_C, sizeof(float) * PQ * PQ * K);
        im2col_gpu_kernel<<<(UNROLL_NB + UNROLL_TPB - 1) / UNROLL_TPB, UNROLL_TPB>>>(PQ * PQ * C,        // num_kernels, = channels * height_col * width_col;
                                                                                     (float *)d_input,   // data_im,
                                                                                     HW,                 // height,
                                                                                     HW,                 // width,
                                                                                     RS,                 // ksize,
                                                                                     0,                  // pad,
                                                                                     STRIDE,             // stride,
                                                                                     PQ,                 // height_col,
                                                                                     PQ,                 // width_col,
                                                                                     (float *)im2col_A); // data_col);

        // printf("Verifying im2col_A: ");
        // float *verification = (float *)malloc(sizeof(float) * RS * RS * PQ * PQ * C);
        // cudaMemcpy(verification, im2col_A, sizeof(float) * RS * RS * PQ * PQ * C, cudaMemcpyDeviceToHost);
        // verify_im2col(verification, 1.0f);

        ker2row_kernel<<<K, C * RS * RS>>>((float(*)[C * RS * RS]) gemm_B,
                                           (float(*)[C][RS][RS])d_weight);

#if GEMM_GLOBAL
        int total = K * PQ * PQ;
        int threadsPerBlock = min(1024, PQ * PQ);
        gemm_global_kernel<<<(total + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>((float(*)[C * RS * RS]) gemm_B, (float(*)[PQ * PQ]) im2col_A,
                                                                                                 (float(*)[PQ * PQ]) d_output);
#else
        int m = K;           // l.n / l.groups
        int k = C * RS * RS; // l.size*l.size
        int n = PQ * PQ;     // l.out_w*l.out_h

        float *a = gemm_B;   // l.weights_gpu + j*l.nweights / l.groups;
        float *b = im2col_A; // state.workspace
        float *c = d_output; // l.output_gpu + (i*l.groups + j)*n*m;

        // gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
        const float alpha = 1, beta = 0;
        cublasHandle_t handle = blas_handle();
        cudaError_t status = (cudaError_t)cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                      n, m, k, &alpha, b, n, a, k, &beta, c, n);
#endif
        cudaFree(im2col_A);
        cudaFree(gemm_B);
        cudaFree(gemm_C);
#endif
    }
    cudaMemcpy(output, d_output, BATCH * PQ * PQ * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    verification(input, weight, output);

    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_input);
}

int main()
{
    float *input = (float *)malloc(sizeof(float) * BATCH * C * HW * HW);
    float *weight = (float *)malloc(sizeof(float) * RS * RS * K * C);
    float *output = (float *)malloc(BATCH * PQ * PQ * K * sizeof(float));

    pass(input, weight, output);

    free(output);
    free(weight);
    free(input);

    return 0;
}

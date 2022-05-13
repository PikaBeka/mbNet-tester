#include <stdio.h>
#include <stdlib.h>

#define IF_N 32
#define INPUT_S 32
#define OF_N 64
#define K_S 5
#define OUTPUT_S (INPUT_S - K_S + 1)

#define TILE_S 8
#define LIM (TILE_S - K_S + 1)

__global__ void convolution(float *input, float *weight, float *output)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    int z = threadIdx.z;
    __shared__ float shm[IF_N][TILE_S][TILE_S];

    int control = z * INPUT_S * INPUT_S + (row + LIM * blockIdx.y) * INPUT_S + col + LIM * blockIdx.x;

    if (control < IF_N * INPUT_S * INPUT_S)
    {
        shm[z][row][col] = input[z * INPUT_S * INPUT_S + (row + LIM * blockIdx.y) * INPUT_S + (col + LIM * blockIdx.x)];
    }
    else
    {
        shm[z][row][col] = 0.0f;
    }

    __syncthreads();

    float temp = 0.0f;

    if (row < LIM && col < LIM && col + LIM * blockIdx.x < OUTPUT_S && row + blockIdx.y * LIM < OUTPUT_S)
    {
        for (int k = 0; k < IF_N; k++)
        {
            for (int i = 0; i < K_S; i++)
            {
                for (int j = 0; j < K_S; j++)
                {
                    temp += shm[k][row + i][col + j] * weight[blockIdx.z * IF_N * K_S * K_S + k * K_S * K_S + i * K_S + j];
                }
            }
        }
        output[blockIdx.z * OUTPUT_S * OUTPUT_S + (row + blockIdx.y * LIM) * OUTPUT_S + (col + LIM * blockIdx.x)] = temp;
    }
}

int main()
{
    float *input = (float *)malloc(sizeof(float) * IF_N * INPUT_S * INPUT_S);
    float *weight = (float *)malloc(sizeof(float) * K_S * K_S * OF_N * IF_N);
    float *output = (float *)malloc(OUTPUT_S * OUTPUT_S * OF_N * sizeof(float));

    srand(time(0));

    for (int i = 0; i < IF_N; i++)
    {
        for (int j = 0; j < INPUT_S; j++)
        {
            for (int k = 0; k < INPUT_S; k++)
            {
                input[i * INPUT_S * INPUT_S + j * INPUT_S + k] = (float)(rand() % 100);
            }
        }
    }

    for (int i = 0; i < OF_N; i++)
    {
        for (int t = 0; t < IF_N; t++)
        {
            for (int j = 0; j < K_S; j++)
            {
                for (int k = 0; k < K_S; k++)
                {
                    weight[i * (IF_N * K_S * K_S) + t * (K_S * K_S) + j * K_S + k] = (float)(rand() % 100);
                }
            }
        }
    }

    float *d_input, *d_weight, *d_output;

    cudaMalloc((void **)&d_input, IF_N * INPUT_S * INPUT_S * sizeof(float));
    cudaMalloc((void **)&d_weight, K_S * K_S * OF_N * IF_N * sizeof(float));
    cudaMalloc((void **)&d_output, OUTPUT_S * OUTPUT_S * OF_N * sizeof(float));

    cudaMemcpy(d_input, input, IF_N * INPUT_S * INPUT_S * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, K_S * K_S * OF_N * IF_N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_S, TILE_S, IF_N);
    dim3 blocks((OUTPUT_S + LIM - 1) / LIM, (OUTPUT_S + LIM - 1) / LIM, OF_N);

    // printf("OUTPUT_S: %d, LIM: %d, dimBlock.x: %d, dimBlock.y: %d\n", OUTPUT_S, LIM, (OUTPUT_S + LIM - 1) / LIM, (OUTPUT_S + LIM - 1) / LIM);

    convolution<<<blocks, threads>>>(d_input, d_weight, d_output);

    cudaMemcpy(output, d_output, OUTPUT_S * OUTPUT_S * OF_N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    float tempC = 0.0f;

    for (int i = 0; i < OF_N; i++)
    {
        for (int j = 0; j < OUTPUT_S; j++)
        {
            for (int k = 0; k < OUTPUT_S; k++)
            {
                tempC = 0.0f;
                for (int l = 0; l < IF_N; l++)
                {
                    for (int m = 0; m < K_S; m++)
                    {
                        for (int t = 0; t < K_S; t++)
                        {
                            tempC += weight[i * IF_N * K_S * K_S + l * K_S * K_S + m * K_S + t] * input[l * INPUT_S * INPUT_S + (j + m) * INPUT_S + (k + t)];
                        }
                    }
                }
                if (output[i * OUTPUT_S * OUTPUT_S + j * OUTPUT_S + k] != tempC)
                {
                    printf("The error is here. The actual result is %f, we get %f on (%d, %d, %d)\n", output[i * OUTPUT_S * OUTPUT_S + j * OUTPUT_S + k], tempC, i, j, k);
                    exit(-1);
                }
            }
        }
    }

    printf("Pointer Tiling convolution finished. It is correct with sizes (%d, %d, %d)\n", IF_N, INPUT_S, OF_N);

    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_input);

    free(output);
    free(weight);
    free(input);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>

#define IF_N 32
#define INPUT_S 32
#define OF_N 64
#define K_S 5
#define OUTPUT_S (INPUT_S - K_S + 1)

__global__ void convolution(float input[IF_N][INPUT_S][INPUT_S], float weight[OF_N][IF_N][K_S][K_S], float output[OF_N][OUTPUT_S][OUTPUT_S])
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int out_ch = idx % OF_N;

    int output_x = (idx / OF_N) % OUTPUT_S;
    int output_y = (idx / OF_N / OUTPUT_S) % OUTPUT_S;

    float tempC = 0.0f;

    for (int k = 0; k < IF_N; k++)
    {
        for (int i = 0; i < K_S; i++)
        {
            for (int j = 0; j < K_S; j++)
            {
                tempC += weight[out_ch][k][i][j] * input[k][output_x + i][output_y + j];
            }
        }
    }

    output[out_ch][output_x][output_y] = tempC;
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

    int threads = min(64, INPUT_S * INPUT_S);
    float total = OF_N * (OUTPUT_S * OUTPUT_S);

    convolution<<<(total + threads - 1) / threads, threads>>>((float(*)[INPUT_S][INPUT_S])d_input, (float(*)[IF_N][K_S][K_S])d_weight, (float(*)[OUTPUT_S][OUTPUT_S])d_output);

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

    printf("Naive convolution finished. It is correct with sizes (%d, %d, %d)\n", IF_N, INPUT_S, OF_N);

    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_input);

    free(output);
    free(weight);
    free(input);

    return 0;
}
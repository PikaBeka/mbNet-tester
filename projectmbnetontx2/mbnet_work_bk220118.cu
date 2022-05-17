#include "slenet_params.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/param.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "mnist.h"
#include "cublas_utils.h"
//#include "utils_user.h"

#include "smbnet.h"
//#include "mblenet5.h"

#define DBG 1

int print_status = 1; 


// Defining the Layer class
class Layer {
	public:
		int M, N, O;
		float *pre_output, *output;
		float *weight, *bias;
    float *im2col_A; //for im2col
    float *gemm_B;
    float *gemm_C;

		Layer(int M, int N, int O);
		~Layer();

    void clear();
};

Layer::Layer(int M, int N, int O) {
	this->M = M;
	this->N = N;
	this->O = O;

	float *temp_weight, *temp_bias;

	// Initializing weights and biases
	temp_weight = (float*)malloc(sizeof(float) * M * N);
	temp_bias = (float*)malloc(sizeof(float) * N);

  temp_weight[0] = 0.0f; 
	for (int i = 0; i < M * N; i++){
		temp_weight[i] = WEIGHT; //1.0f;
    //temp_weight[i+1] = temp_weight[i]+1.0f; 
  }

	for (int i = 0; i < N; i++)
		temp_bias[i] = BIAS; //1.0f;

	// Allocating space for CUDA variables
	cudaMalloc(&pre_output, sizeof(float) * O);
	cudaMalloc(&output, sizeof(float) * O);
	cudaMalloc(&weight, sizeof(float) * M * N);
	cudaMalloc(&bias, sizeof(float) * N);

  cudaMalloc(&im2col_A, sizeof(float) *M*O/N);
  cudaMalloc(&gemm_B, sizeof(float) *M*N);
  cudaMalloc(&gemm_C, sizeof(float) *(O/N)*N);

	// Copying weights and biases to CUDA variables
	cudaMemcpy(weight, temp_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(bias, temp_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

	// Freeing temporary weights and biases
	free(temp_weight);
	free(temp_bias);
}

Layer::~Layer() {
	// Freeing all CUDA varibles of a layer
	cudaFree(pre_output);
	cudaFree(output);
	cudaFree(weight);
	cudaFree(bias);
  cudaFree(im2col_A);
}

void Layer::clear(){
	cudaMemset(pre_output, 0x00, sizeof(float)*O); 
	cudaMemset(output, 0x00, sizeof(float)*O); 
}

// Initializing a convolutional layer
Layer conv_layer(FILTER_SIZE * FILTER_SIZE, CHANNEL, CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
Layer ss_layer(SS_SIZE * SS_SIZE, SS_CHANNELS, CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
Layer fc_layer(CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, NUM_CLASSES, NUM_CLASSES);

#if LENET5_C2
Layer conv2_layer(FILTER_SIZE2 * FILTER_SIZE2, CHANNEL2, CHANNEL2 * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
#endif 
#if LENET5_SS2
Layer ss2_layer(SS2_SIZE * SS2_SIZE, SS2_CHANNELS, CHANNEL2 * SS2_OUTPUT_SIZE * SS2_OUTPUT_SIZE);
#endif 

double time_taken = 0.0;

#if CONV_COMPOSITE
__global__ void kernel_conv1_composite(float input[INCH][INSIZE][INSIZE], 
                                      float pre_output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], 
                                      float weight[CHANNEL][FILTER_SIZE][FILTER_SIZE],
                                      float bias[CHANNEL], 
                                      float output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE]) {

  #if CONV_SHARED
	int tidx = threadIdx.x;
	int bIdx = blockIdx.x;
	
	__shared__ float sh_img[SHBS][SHBS];
	__shared__ float sh_weight[CHANNEL][FILTER_SIZE][FILTER_SIZE];

	#if 1
	int img_row = tidx / SHBS; 
	int img_col = tidx % SHBS; 
	#else 
	int img_row = (itemp/=1) % SHBS;
	int img_col = (itemp/=SHBS) % SHBS;
	#endif 

	#if 1
	int bIdx_r = bIdx / GRID; 
	int bIdx_c = bIdx % GRID; 
	#else 
	int itemp = bIdx; 
	int bIdx_r = (itemp/=1) % GRID;
	int bIdx_c = (itemp/=GRID) % GRID;
	#endif 

	/* input image copy to shared memory */ 
	if (tidx < SHBS*SHBS){
		//sh_img[img_row][img_col] = 0; 
		sh_img[img_row][img_col] = input[blockIdx.y][bIdx_r*BS + img_row][bIdx_c*BS + img_col];
	}
  	__syncthreads();
	

	#if 1
	int ch = tidx / (FILTER_SIZE*FILTER_SIZE);
	int k_row = (tidx % (FILTER_SIZE*FILTER_SIZE)) / FILTER_SIZE;
	int k_col = (tidx % (FILTER_SIZE*FILTER_SIZE)) % FILTER_SIZE;
	#else 
	itemp = tidx;
	int ftr = (itemp/=1)%CHANNEL; 
	int k_row = (itemp/=CHANNEL) % FILTER_SIZE;
	int k_col = (itemp/=FILTER_SIZE) % FILTER_SIZE;
	#endif 
	
	/* kernel filter copy to shared memory */ 
	if (tidx < CHANNEL*FILTER_SIZE*FILTER_SIZE){
		//sh_weight[ch][k_row][k_col] = 0; 
		sh_weight[ch][k_row][k_col] = weight[ch][k_row][k_col];
	}
	__syncthreads();

	ch = tidx/(BS*BS);
	int w_row = (tidx % (BS*BS)) / BS;
	int w_col = (tidx % (BS*BS)) % BS; 

	float sum = 0;
	if (w_row < BS && w_col < BS && ch < CHANNEL) {
		for (int i = 0; i < FILTER_SIZE; i++)
			for (int j = 0; j < FILTER_SIZE; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ch][i][j];
		pre_output[ch][bIdx_r * BS + w_row][bIdx_c * BS + w_col] = sum;
    	pre_output[ch][bIdx_r * BS + w_row][bIdx_c * BS + w_col] += bias[ch];
    	output[ch][bIdx_r * BS + w_row][bIdx_c * BS + w_col] = 1 / (1 + exp(-pre_output[ch][bIdx_r * BS + w_row][bIdx_c *  BS + w_col]));    
  }

  #else
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
	float tempC = 0.0f;

	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			tempC += weight[channel][i][j] * input[i + output_x][j + output_y];
		}
	}
  if (idx < CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE){
	    pre_output[channel][output_x][output_y] = tempC;
      pre_output[channel][output_x][output_y] += bias[channel];
      output[channel][output_x][output_y] = 1 / (1 + exp(-pre_output[channel][output_x][output_y]));
    }

  #endif 
}
#else

__global__ void kernel_conv_filter(float input[INCH][INSIZE][INSIZE], 
                                  float pre_output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], 
                                  float weight[CHANNEL][FILTER_SIZE][FILTER_SIZE]) {
	#if CONV_SHARED
	int tidx = threadIdx.x;
	int bIdx = blockIdx.x;
	
	__shared__ float sh_img[SHBS][SHBS];
	__shared__ float sh_weight[CHANNEL][FILTER_SIZE][FILTER_SIZE];

	#if 1
	int img_row = tidx / SHBS; 
	int img_col = tidx % SHBS; 
	#else 
	int img_row = (itemp/=1) % SHBS;
	int img_col = (itemp/=SHBS) % SHBS;
	#endif 

	#if 1
	int bIdx_r = bIdx / GRID; 
	int bIdx_c = bIdx % GRID; 
	#else 
	int itemp = bIdx; 
	int bIdx_r = (itemp/=1) % GRID;
	int bIdx_c = (itemp/=GRID) % GRID;
	#endif 

	/* input image copy to shared memory */ 
	if (tidx < SHBS*SHBS){
		//sh_img[img_row][img_col] = 0; 
		sh_img[img_row][img_col] = input[blockIdx.y][bIdx_r*BS + img_row][bIdx_c*BS + img_col];
	}
  	__syncthreads();
	

	#if 1
	int ch = tidx / (FILTER_SIZE*FILTER_SIZE);
	int k_row = (tidx % (FILTER_SIZE*FILTER_SIZE)) / FILTER_SIZE;
	int k_col = (tidx % (FILTER_SIZE*FILTER_SIZE)) % FILTER_SIZE;
	#else 
	itemp = tidx;
	int ftr = (itemp/=1)%CHANNEL; 
	int k_row = (itemp/=CHANNEL) % FILTER_SIZE;
	int k_col = (itemp/=FILTER_SIZE) % FILTER_SIZE;
	#endif 
	
	/* kernel filter copy to shared memory */ 
	if (tidx < CHANNEL*FILTER_SIZE*FILTER_SIZE){
		//sh_weight[ch][k_row][k_col] = 0; 
		sh_weight[ch][k_row][k_col] = weight[ch][k_row][k_col];
	}
	__syncthreads();

	ch = tidx/(BS*BS);
	int w_row = (tidx % (BS*BS)) / BS;
	int w_col = (tidx % (BS*BS)) % BS; 

	float sum = 0;
	if (w_row < BS && w_col < BS && ch < CHANNEL) {
		for (int i = 0; i < FILTER_SIZE; i++)
			for (int j = 0; j < FILTER_SIZE; j++)
				sum += sh_img[w_row + i][w_col + j] * sh_weight[ch][i][j];
		pre_output[ch][bIdx_r * BS + w_row][bIdx_c * BS + w_col] = sum;
	}	

	#else
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
	
	float tempC = 0.0f;
	for (int i = 0; i < FILTER_SIZE; i++) {
		for (int j = 0; j < FILTER_SIZE; j++) {
			tempC += weight[channel][i][j] * input[i + output_x][j + output_y];
		}
	}
  	if (idx < CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE)
		pre_output[channel][output_x][output_y] = tempC;
	#endif 
}
__global__ void kernel_conv_bias(float pre_output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float bias[CHANNEL]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
  if (idx < CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE)
	  pre_output[channel][output_x][output_y] += bias[channel];
}

__global__ void kernel_conv_sigmoid(float preact[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], 
                          float output[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
  if (idx < CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE)
	  output[channel][output_x][output_y] = 1 / (1 + exp(-preact[channel][output_x][output_y]));
}
#endif 

#if SS_COMPOSITE
__global__ void kernel_ss1_composite(float input[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE],
#else
__global__ void kernel_ss1_filter(float input[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE],
#endif  
								float pre_output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], 
								float weight[SS_CHANNELS][SS_SIZE][SS_SIZE],
								float bias[SS_CHANNELS], float output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % SS_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / SS_OUTPUT_SIZE) % SS_OUTPUT_SIZE;
	float tempC = 0.0f;

	for (int i = 0; i < SS_SIZE; i++) {
		for (int j = 0; j < SS_SIZE; j++) {
			tempC += weight[0][i][j] * input[channel][i + output_x * SS_STRIDE][j + output_y * SS_STRIDE];
		}
	}
  	if (idx < CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE){
		pre_output[channel][output_x][output_y] = tempC;
		#if SS_COMPOSITE
		pre_output[channel][output_x][output_y] += bias[0];
		output[channel][output_x][output_y] = 1 / (1 + exp(-pre_output[channel][output_x][output_y]));
		#endif
	}
}

__global__ void kernel_ss1_bias(float pre_output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], float bias[SS_CHANNELS]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % SS_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / SS_OUTPUT_SIZE) % SS_OUTPUT_SIZE;
  if (idx < CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE)
	  pre_output[channel][output_x][output_y] += bias[0];
}

__global__ void kernel_ss1_sigmoid(float pre_output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], float output[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % CHANNEL;
	int output_x = (idx / CHANNEL) % SS_OUTPUT_SIZE;
	int output_y = (idx / CHANNEL / SS_OUTPUT_SIZE) % SS_OUTPUT_SIZE;
  if (idx < CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE)
	  output[channel][output_x][output_y] = 1 / (1 + exp(-pre_output[channel][output_x][output_y]));
}

#if FC_COMPOSITE
__global__ void kernel_fc1_composite(float input[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], 
#else 
__global__ void kernel_fc1(float input[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE], 
#endif 
							float pre_output[NUM_CLASSES], 
                            float weight[NUM_CLASSES][CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE],
							float bias[NUM_CLASSES], float output[NUM_CLASSES]) {

  	#if FC_SHARED
  	//int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidx = threadIdx.x; 	  
	int bidx = blockIdx.x; 
  	__shared__ float tempA[NUM_CLASSES][CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE]; 

  	int ch = tidx  % CHANNEL;
  	int x = (tidx /  CHANNEL) % SS_OUTPUT_SIZE;
  	int y = (tidx / CHANNEL / SS_OUTPUT_SIZE) % SS_OUTPUT_SIZE;

  	float tempC = 0.0f; 
  	if (tidx < CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE){
		//input[ch][x][y] = 1.0f; 
  		tempA[blockIdx.x][ch][x][y] =  weight[blockIdx.x][ch][x][y] * input[ch][x][y];
		__syncthreads(); 
	}

	if (tidx  == (blockDim.x-1)){
		for(int i=0; i<CHANNEL ; i++)
			for(int j=0; j<SS_OUTPUT_SIZE ; j++)
				for(int k=0; k<SS_OUTPUT_SIZE ; k++)
					tempC += tempA[blockIdx.x][i][j][k]; 
		pre_output[blockIdx.x] = tempC;
		#if FC_COMPOSITE
		pre_output[bidx] += bias[bidx];
		output[bidx] = 1 / (1 + exp(-pre_output[bidx]));
		#endif 
  	}

 	#else
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % NUM_CLASSES;
	float tempC = 0.0f;

  	if (idx < NUM_CLASSES){
		for (int i = 0; i < CHANNEL; i++) {
			for (int j = 0; j < SS_OUTPUT_SIZE; j++) {
				for (int k = 0; k < SS_OUTPUT_SIZE; k++) {
					tempC += weight[channel][i][j][k] * input[i][j][k];
				}
			}
		}
		pre_output[channel] = tempC;
		#if FC_COMPOSITE
		pre_output[channel] += bias[channel];
		output[channel] = 1 / (1 + exp(-pre_output[channel]));
		#endif 
	}
  #endif                           
}

__global__ void kernel_fc1_bias(float pre_output[NUM_CLASSES], float bias[NUM_CLASSES]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % NUM_CLASSES;
	pre_output[channel] += bias[channel];
}

__global__ void kernel_fc1_sigmoid(float pre_output[NUM_CLASSES], float output[NUM_CLASSES]) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int channel = idx % NUM_CLASSES;
	output[channel] = 1 / (1 + exp(-pre_output[channel]));
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
__global__ void ker2row_kernel(float weight_col[CHANNEL][FILTER_SIZE*FILTER_SIZE], float weight[CHANNEL][FILTER_SIZE][FILTER_SIZE]) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  
  int channel = idx % CHANNEL;
	int x = (idx / CHANNEL) % FILTER_SIZE;
	int y = (idx / CHANNEL / FILTER_SIZE) % FILTER_SIZE;
  if (idx < CHANNEL*FILTER_SIZE*FILTER_SIZE)
	  weight_col[channel][x*FILTER_SIZE + y] = weight[channel][x][y];
}

__global__ void gemm_global_kernel(float matB[CHANNEL][FILTER_SIZE*FILTER_SIZE] 
                      ,float matA[FILTER_SIZE*FILTER_SIZE][CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE]
                      ,float matC[CHANNEL][CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE]) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x; 

  int x = idx % CHANNEL;
	int y = (idx / CHANNEL) % (CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);

  float tempC; 
  //matC[x][y] = 0.0f; 
  if (idx < CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE){
    for(int i=0; i<FILTER_SIZE*FILTER_SIZE; i++){
      tempC += matB[x][i] * matA[i][y]; 
    } 
    matC[x][y] = tempC; 
  }
}

__global__ void col2im_kernel(float preout[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float preout_col[CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE][CHANNEL]) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  
  int channel = idx % CHANNEL;
	int x = (idx / CHANNEL) % CONV_OUTPUT_SIZE;
	int y = (idx / CHANNEL / CONV_OUTPUT_SIZE) % CONV_OUTPUT_SIZE;
  if (idx < CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE)
	  preout[channel][x][y] = preout_col[x*FILTER_SIZE + y][channel];
}

///*
void verifyConv(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0; 
	for (int i = 0; i < CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE; i++){ 
		maxError = max(abs(A[i] - val), maxError);
    if (maxError != 0)
      cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE);
}

#if LENET5_C2
void verifyConv2(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0; 
	for (int i = 0; i < CHANNEL2 * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE; i++){ 
		maxError = max(abs(A[i] - val), maxError);
		//printf("%.1f ", maxError); 
    if (maxError > 0.2)
      cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, CHANNEL2*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE);
}
#endif 


#if 1
void debugPrintConv(float A[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE]){
  //if (print_status == 1){
  
      for(int i=0; i<CHANNEL; i++){
        for(int j=0; j<CONV_OUTPUT_SIZE; j++){
          for(int k=0; k<CONV_OUTPUT_SIZE ; k++){
            printf("%3.2f ", A[i][j][k]);
          }
          printf("\n");
        }   
        printf("\n");
    }
   // print_status--; 
    printf("\n\n");
  //}
}
#endif 

void verify_im2col(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0; 
	for (int i = 0; i < FILTER_SIZE * FILTER_SIZE * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE; i++){ 
		maxError = max(abs(A[i] - val), maxError);
    if (maxError != 0)
      cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, FILTER_SIZE * FILTER_SIZE *CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE);
}


void verify_ker2row(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0; 
	for (int i = 0; i < CHANNEL * FILTER_SIZE * FILTER_SIZE; i++){ 
		maxError = max(abs(A[i] - val), maxError);
    if (maxError != 0)
      cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, CHANNEL*FILTER_SIZE*FILTER_SIZE);
}

void verifySS(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0;
	for (int i = 0; i < CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE; i++){
		maxError = max(abs(A[i] - val), maxError);
    	if (maxError > 0.007f)
      		cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE);
}

#if LENET5_SS2
void verifySS2(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0;
	for (int i = 0; i < CHANNEL2 * SS2_OUTPUT_SIZE * SS2_OUTPUT_SIZE; i++){
		maxError = max(abs(A[i] - val), maxError);
    	if (maxError > 0.007f)
      		cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, CHANNEL2*SS2_OUTPUT_SIZE*SS2_OUTPUT_SIZE);
}
#endif 

void verifyFCfil(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0;
	for (int i = 0; i < NUM_CLASSES; i++){
		maxError = max(abs(A[i] - val), maxError);
    	if (maxError > 4.0f)
      		cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, NUM_CLASSES);
}


void verifyFC(float *A, float val) {
	float maxError = 0.0f;

  int cnt = 0;
	for (int i = 0; i < NUM_CLASSES; i++){
		maxError = max(abs(A[i] - val), maxError);
    if (maxError > 0.0009f)
      cnt++; 
  }
	printf("maxError = %f (cnt = %d),%d)\n", maxError, cnt, NUM_CLASSES);
}
//*/
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {

    int index = blockIdx.x*blockDim.x+threadIdx.x;

    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                //data_im[(channel_in * height + h_in) * width + w_in + i * width + j];
                //(*data_col_ptr) = data_im_ptr[ii * width + jj];

                data_col_ptr += height_col * width_col;
            }
        }
    }
}


// Performing a forward pass using a single image
static double forward_pass(double data[INSIZE][INSIZE], bool verify) {
	// Copying a double data to a float data
	float input[INSIZE][INSIZE];
	float *verification;
  
  input[0][0] = 0.0f; 
	for (int i = 0; i < INSIZE; i++) {
		for (int j = 0; j < INSIZE; j++){
			#if SIMULATION
      		input[i][j] = INPUT; //Simulated data
			#else
			input[i][j] = data[i][j];  //MNIST data
			#endif 
      //printf("%f ",input[i][j]);
    }
    //printf("\n");
	}

//Layer conv_layer(FILTER_SIZE * FILTER_SIZE, CHANNEL, CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
//Layer ss_layer(SS_SIZE * SS_SIZE, SS_CHANNELS, CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
//Layer fc_layer(CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, NUM_CLASSES, NUM_CLASSES);

  //l_input.clear(); 
	//conv_layer.clear(); //need to be checked !!!
	//ss_layer.clear();
	//fc_layer.clear();

	float (*d_input)[INSIZE][INSIZE];
	cudaMalloc(&d_input, sizeof(float) * INSIZE * INSIZE * INCH);
	cudaMemcpy(d_input, input, sizeof(float) * INSIZE * INSIZE * INCH, cudaMemcpyHostToDevice);

  //For im2col workspace 
  //float* im2col_input;  
  //cudaMalloc(&im2col_input, sizeof(float) * INSIZE * INSIZE);

  //float* im2col_workspace;  
  //cudaMalloc(&im2col_workspace, sizeof(float) * FILTER_SIZE * FILTER_SIZE * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

#if CPU_GEMM
float matB[CHANNEL][FILTER_SIZE * FILTER_SIZE];
float matA[FILTER_SIZE * FILTER_SIZE][CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE];
float matC[CHANNEL][CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE];
#endif 
//float newMatC[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE];
//float gemmMatC[CHANNEL][CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE];

#if CONV_COMPOSITE
	// Performing Convolutional composite
	#if CONV_SHARED
	const dim3 numBlocks(CONV_NB,1);
  	const dim3 threadsPerBlock(CONV_TPB);
	kernel_conv1_composite<<<numBlocks, threadsPerBlock>>>(d_input, 
	#else
	kernel_conv1_composite<<<(N1+K1-1)/K1, K1>>>(d_input, 
	#endif  
                                            (float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output,
                                            (float(*)[FILTER_SIZE][FILTER_SIZE])conv_layer.weight,
                                            conv_layer.bias, 
                                            (float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.output);

	// Verifying Convolutional composite
	if (verify) {
		printf("Veri Convolutional composite: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
		cudaMemcpy(verification, conv_layer.pre_output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifyConv(verification, INPUT*WEIGHT*FILTER_SIZE*FILTER_SIZE + BIAS); //25.0f
		free(verification);
	}

#elif DIRECT 
	// Performing Convolutional filtering
  	#if CONV_SHARED
	const dim3 numBlocks(CONV_NB,1);
  	const dim3 threadsPerBlock(CONV_TPB);
	kernel_conv_filter<<<numBlocks, threadsPerBlock>>>(d_input, 
	#else
	kernel_conv_filter<<<(N1+K1-1)/K1, K1>>>(d_input,
	#endif  
                                            (float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output,
                                            (float(*)[FILTER_SIZE][FILTER_SIZE])conv_layer.weight);

	// Verifying Convolutional filtering operation
	if (verify) {
		printf("Veri Convolutional filtering: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
		cudaMemcpy(verification, conv_layer.pre_output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifyConv(verification, INPUT*WEIGHT*FILTER_SIZE*FILTER_SIZE); //25.0f
		free(verification);
	}

#else // GEMM // gemm or direct setting 
  //im2col_gpu_kernel_ext<<<(N1+K1-1)/K1, K1>>>(CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE, d_input, INSIZE, INSIZE, FILTER_SIZE, FILTER_SIZE, 0, 0, STRIDE, STRIDE, 1, 1, CONV_OUTPUT_SIZE, CONV_OUTPUT_SIZE,ic_workspace);
///*
  im2col_gpu_kernel<<<(N11+K11-1)/K11, K11>>>(CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE, //num_kernels, = channels * height_col * width_col; 
                                              (float *)d_input, //data_im, 
                                              INSIZE, //height, 
                                              INSIZE, //width, 
                                              FILTER_SIZE, //ksize, 
                                              0, //pad,
                                              STRIDE, //stride, 
                                              CONV_OUTPUT_SIZE, //height_col, 
                                              CONV_OUTPUT_SIZE, //width_col, 
                                              (float *)conv_layer.im2col_A); //data_col);

  	// Verifying im2col operation
	if (verify) { //verify
    if (print_status == 1){
		printf("Verifying im2col_A: ");
		verification = (float*)malloc(sizeof(float) * FILTER_SIZE * FILTER_SIZE * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
		cudaMemcpy(verification, conv_layer.im2col_A, sizeof(float) * FILTER_SIZE * FILTER_SIZE * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verify_im2col(verification, INPUT); //-1.0f 

    #if 0
    for(int i=0; i<INSIZE*INSIZE; i++){ 
      
      if (i%(INSIZE) == 0){
           printf("\n");
      }
      printf("%2.1f ", verification[i]);
    }
    printf("\n");
    #endif 
    free(verification);
    print_status--;
    }
  }                                     
		
  //ker2col operation 
  ker2row_kernel<<<CHANNEL, FILTER_SIZE * FILTER_SIZE>>>((float(*)[FILTER_SIZE*FILTER_SIZE])conv_layer.gemm_B, 
                                                    (float(*)[FILTER_SIZE][FILTER_SIZE])conv_layer.weight); 
// Verifying ker2row operation
	if (verify) { //verify
		printf("Verifying ker2row_A: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * FILTER_SIZE * FILTER_SIZE);
		cudaMemcpy(verification, conv_layer.gemm_B, sizeof(float) * CHANNEL * FILTER_SIZE * FILTER_SIZE, cudaMemcpyDeviceToHost);
		verify_ker2row(verification, WEIGHT); //-1.0f 
		free(verification); 
		print_status--; 
	}

///* //on cpu gemm 	
#if CPU_GEMM 
  //float matA[FILTER_SIZE * FILTER_SIZE][CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE];
  cudaMemcpy(matA, conv_layer.im2col_A, sizeof(float) * FILTER_SIZE * FILTER_SIZE * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
  //float matB[CHANNEL][FILTER_SIZE * FILTER_SIZE]; 
  cudaMemcpy(matB, conv_layer.gemm_B, sizeof(float) * FILTER_SIZE * FILTER_SIZE * CHANNEL, cudaMemcpyDeviceToHost);
 
  #if 1
  //float matC[CHANNEL][CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE]; 
  //gemm_custom_cpu(); 
  {

    for(int i=0; i<CHANNEL ; i++){
      for(int j=0; j< CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE ; j++){
        matC[i][j] = 0.0f;
        for(int k=0; k< FILTER_SIZE*FILTER_SIZE; k++){
          matC[i][j] += matB[i][k] * matA[k][j]; 
        }
      }
    }
  }
  #endif 

	//kernel_conv_filter<<<(N1+K1-1)/K1, K1>>>(d_input, 
  //                                          (float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output,
  //                                          (float(*)[FILTER_SIZE][FILTER_SIZE])conv_layer.weight);

  cudaMemcpy(conv_layer.pre_output, matC, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyHostToDevice);
  //cudaMemcpy(conv_layer.pre_output, newMatC, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyHostToDevice); //both are okay

#elif GEMM_GLOBAL// GPU_GEMM 

  gemm_global_kernel<<<CHANNEL, CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE>>>((float(*)[FILTER_SIZE*FILTER_SIZE])conv_layer.gemm_B 
                      ,(float(*)[CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE])conv_layer.im2col_A
                      //,(float(*)[CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE])conv_layer.gemm_C); 
                      ,(float(*)[CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE])conv_layer.pre_output);


#else  //using cublasSgemm

  int m = CHANNEL; // l.n / l.groups
  int k = FILTER_SIZE * FILTER_SIZE; // l.size*l.size
  int n = CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE; //l.out_w*l.out_h

  float *a = conv_layer.gemm_B; //l.weights_gpu + j*l.nweights / l.groups;
  float *b = conv_layer.im2col_A; //state.workspace
  float *c = conv_layer.pre_output; //l.output_gpu + (i*l.groups + j)*n*m;

  //gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n); 
  const float alpha = 1, beta = 0;
  cublasHandle_t handle = blas_handle();
  cudaError_t status = (cudaError_t)cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         n, m, k, &alpha, b, n, a, k, &beta, c, n);
  //cublasDestroy(handle);

	// Verifying cublasSgemm operation
	if (verify) {
		printf("Veri cublasSgemm: ");
		verification = (float*)malloc(sizeof(float) * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE * CHANNEL);
		cudaMemcpy(verification, conv_layer.pre_output, sizeof(float) * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE * CHANNEL, cudaMemcpyDeviceToHost);
		verifyConv(verification, INPUT*WEIGHT*FILTER_SIZE*FILTER_SIZE); //25.0f
		free(verification);
	}

  #if 0
  if (print_status == 1){
    //verification = (float*)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
    cudaMemcpy(newMatC, conv_layer.pre_output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
  
      for(int i=0; i<1; i++){ //CHANNEL
        for(int j=0; j<CONV_OUTPUT_SIZE; j++){
          for(int k=0; k<CONV_OUTPUT_SIZE ; k++){
            printf("%3.1f ", newMatC[i][j][k]);
          }
          printf("\n");
        }   
        printf("\n");
    }
    print_status--; 
    printf("\n\n");
  }
  #endif 

#endif //cublasSgemm 
#endif //GEMM


#if !CONV_COMPOSITE
	// Performing Convolutional bias addition
	kernel_conv_bias<<<(N1+K1-1)/K1, K1>>>((float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output, conv_layer.bias);

	// Verifying Convolutional bias operation
	if (verify) {
		printf("Veri Convolutional bias: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
		cudaMemcpy(verification, conv_layer.pre_output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifyConv(verification, INPUT*WEIGHT*FILTER_SIZE*FILTER_SIZE + BIAS); // 26.0f
		free(verification);
	}

	// Performing Convolutional sigmoid operation
	kernel_conv_sigmoid<<<(N1+K1-1)/K1, K1>>>((float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.pre_output, (float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.output);

	// Verifying Convolutional sigmoid operation
	if (verify) {
		printf("Veri Convolutional sigmoid: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE);
		cudaMemcpy(verification, conv_layer.output, sizeof(float) * CHANNEL * CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifyConv(verification, CONV_POST_ACT); // 1.0f
		free(verification);
	}
#endif //!CONV_COMPOSITE

#if ALL_LAYERS 

	// Performing Subsampling filtering
	#if SS_COMPOSITE 
	kernel_ss1_composite<<<(N2+K2-1)/K2, K2>>>((float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.output, 
	#else
	kernel_ss1_filter<<<(N2+K2-1)/K2, K2>>>((float(*)[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE])conv_layer.output, 
	#endif 
											(float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.pre_output, 
											(float(*)[SS_SIZE][SS_SIZE])ss_layer.weight, 
											ss_layer.bias, (float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.output);

	// Verifying Subsampling filtering operation
	if (verify) {
		printf("Veri Subsampling filtering: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
		#if SS_COMPOSITE
		cudaMemcpy(verification, ss_layer.output, sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
        verifySS(verification, SS_POST_ACT); //1.0f
		#else
		cudaMemcpy(verification, ss_layer.pre_output, sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifySS(verification, CONV_POST_ACT*WEIGHT*SS_SIZE*SS_SIZE); //16.0f
		#endif 
		free(verification);
	}

	#if !SS_COMPOSITE
	// Performing Subsampling bias addition
	kernel_ss1_bias<<<(N2+K2-1)/K2, K2>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.pre_output, ss_layer.bias);

	// Verifying Subsampling bias operation
	if (verify) {
		printf("Veri Subsampling bias: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
		cudaMemcpy(verification, ss_layer.pre_output, sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifySS(verification, CONV_POST_ACT*WEIGHT*SS_SIZE*SS_SIZE + BIAS); // 17.0f
		free(verification);
	}

	// // Performing Subsampling sigmoid operation
	kernel_ss1_sigmoid<<<(N2+K2-1)/K2, K2>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.pre_output, (float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.output);

	// Verifying Subsampling sigmoid operation
	if (verify) {
		printf("Veri Subsampling sigmoid: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE);
		cudaMemcpy(verification, ss_layer.output, sizeof(float) * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifySS(verification, SS_POST_ACT); //1.0f
		free(verification);
	}
	#endif 

	//Layer conv2_layer(FILTER_SIZE2 * FILTER_SIZE2, CHANNEL2, CHANNEL2 * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
	#if LENET5_C2
	const dim3 numBlocks2(CONV_NB2,INNR);
  	const dim3 threadsPerBlock2(CONV_TPB2);
	kernel_conv2_filter<<<numBlocks2, threadsPerBlock2>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.output, //d_input, 
  	
	//kernel_conv2_filter<<<(N1+K1-1)/K1, K1>>>(d_input,
	 
                                            (float(*)[CONV2_OUTPUT_SIZE][CONV2_OUTPUT_SIZE])conv2_layer.pre_output,
                                            (float(*)[FILTER_SIZE2][FILTER_SIZE2])conv2_layer.weight,
											conv2_layer.bias, (float(*)[CONV2_OUTPUT_SIZE][CONV2_OUTPUT_SIZE])conv2_layer.output);			


	// Verifying Convolutional filtering operation
	if (verify) {
		printf("Veri Convolutional 2 filtering: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL2 * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
		cudaMemcpy(verification, conv2_layer.pre_output, sizeof(float) * CHANNEL2 * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifyConv2(verification, SS_POST_ACT*WEIGHT*FILTER_SIZE2*FILTER_SIZE2 + BIAS); // 24.8  25.0f
		
		printf("Veri Convolutional 2 activation: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL2 * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
		cudaMemcpy(verification, conv2_layer.output, sizeof(float) * CHANNEL2 * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifyConv2(verification, CONV2_POST_ACT); // 1.0f 
		
		free(verification);
	}
	#endif 

	#if LENET5_SS2
	kernel_ss2_composite<<<(N22+K22-1)/K22, K22>>>((float(*)[CONV2_OUTPUT_SIZE][CONV2_OUTPUT_SIZE])conv2_layer.output, 
											(float(*)[SS2_OUTPUT_SIZE][SS2_OUTPUT_SIZE])ss2_layer.pre_output, 
											(float(*)[SS2_SIZE][SS2_SIZE])ss2_layer.weight, 
											ss_layer.bias, (float(*)[SS2_OUTPUT_SIZE][SS2_OUTPUT_SIZE])ss2_layer.output);

	// Verifying Subsampling filtering operation
	if (verify) {
		printf("Veri Subsampling2 filtering: ");
		verification = (float*)malloc(sizeof(float) * CHANNEL2 * SS2_OUTPUT_SIZE * SS2_OUTPUT_SIZE);
		cudaMemcpy(verification, ss2_layer.pre_output, sizeof(float) * CHANNEL2 * SS2_OUTPUT_SIZE * SS2_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
		verifySS2(verification, CONV2_POST_ACT*WEIGHT*SS2_SIZE*SS2_SIZE + BIAS); //16.0f
	
		printf("Veri Subsampling2 activation: ");
		cudaMemcpy(verification, ss2_layer.output, sizeof(float) * CHANNEL2 * SS2_OUTPUT_SIZE * SS2_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
        verifySS2(verification, SS2_POST_ACT); //1.0f
		free(verification);
	}

	#endif 

#if LENET5_FC 
	// Performing Fully-Connected Computation
	kernel_fc1_lenet5_composite<<<(N33+K33-1)/K33, K33>>>((float(*)[SS2_OUTPUT_SIZE][SS2_OUTPUT_SIZE])ss2_layer.output,
									(float(*))fc_layer.pre_output, 
									(float(*)[CHANNEL2][SS2_OUTPUT_SIZE][SS2_OUTPUT_SIZE])fc_layer.weight,
									fc_layer.bias, (float(*))fc_layer.output);

	// Verifying Fully-Connected Computation
	if (verify) {
		printf("Veri Fully-Connected Lenet: ");
		verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
		cudaMemcpy(verification, fc_layer.pre_output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
        verifyFCfil(verification, SS2_POST_ACT*WEIGHT*CHANNEL2*SS2_OUTPUT_SIZE*SS2_OUTPUT_SIZE + BIAS);//216.0f
		
		printf("Veri Fully-Connected Lenet activation: ");
		verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
		cudaMemcpy(verification, fc_layer.output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
		verifyFC(verification, FC_POST_ACT);//1.0f

		free(verification);
	}

#else 
	#if FC_COMPOSITE
	// Performing Fully-Connected Computation
	kernel_fc1_composite<<<(N3+K3-1)/K3, K3>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.output,
	#else
	kernel_fc1<<<(N3+K3-1)/K3, K3>>>((float(*)[SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])ss_layer.output,
	#endif  
									(float(*))fc_layer.pre_output, 
									(float(*)[CHANNEL][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE])fc_layer.weight,
									fc_layer.bias, (float(*))fc_layer.output);

	// Verifying Fully-Connected Computation
	if (verify) {
		printf("Veri Fully-Connected: ");
		verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
		#if FC_COMPOSITE
		cudaMemcpy(verification, fc_layer.output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
		verifyFC(verification, FC_POST_ACT);//1.0f
		#else
		cudaMemcpy(verification, fc_layer.pre_output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
        verifyFCfil(verification, SS_POST_ACT*WEIGHT*CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE);//216.0f
		#endif 
		free(verification);
	}

	#if !FC_COMPOSITE	
	// Performing Fully-Connected bias operation
	kernel_fc1_bias<<<1, K3>>>((float(*))fc_layer.pre_output, fc_layer.bias);

	// Verifying Fully-Connected bias operation
	if (verify) {
		printf("Veri Fully-Connected bias: ");
		verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
		cudaMemcpy(verification, fc_layer.pre_output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
		verifyFCfil(verification, SS_POST_ACT*WEIGHT*CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE + BIAS); //217.0f
		free(verification);
	}

	// Performing Fully-Connected sigmoid operation
	kernel_fc1_sigmoid<<<1, K3>>>((float(*))fc_layer.pre_output, (float(*))fc_layer.output);

	// Verifying Fully-Connected sigmoid operation
	if (verify) {
		printf("Veri Fully-Connected sigmoid: ");
		verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
		cudaMemcpy(verification, fc_layer.output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
		verifyFC(verification, FC_POST_ACT); // 1.0f
		free(verification);
	}
	#endif // !FC_COMPOSITE

#endif //LENET5_FC
#endif //ALL_LAYERS 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);


  #if 0
  if (print_status == 1 || print_status == 0 || print_status == -1 || print_status == -2 || print_status == -3 || print_status == -4){
    verification = (float*)malloc(sizeof(float) * NUM_CLASSES);
    cudaMemcpy(verification, fc_layer.output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);
  
      for(int i=0; i<NUM_CLASSES; i++){
        printf("%3.1f[%d] ", verification[i], i);
    }

    print_status--; 
    
    printf("\n");
    free(verification);
  }
  #endif 

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_input);
	//printf("\n"); 
	return elapsedTime;
}

void copy_trained_parameters() {
	cudaMemcpy(conv_layer.weight, c1_weight, sizeof(float) * CHANNEL * FILTER_SIZE * FILTER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(conv_layer.bias, c1_bias, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(ss_layer.weight, s2_weight, sizeof(float) * SS_SIZE * SS_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(ss_layer.bias, s2_bias, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fc_layer.weight, f3_weight, sizeof(float) * NUM_CLASSES * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(fc_layer.bias, f3_bias, sizeof(float) * NUM_CLASSES, cudaMemcpyHostToDevice);
}

int main() {
	int ret, i;
	mnist_data *test_set;
	static unsigned int test_cnt;

	// Calling the mnist_load() function
	if (ret = mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", &test_set, &test_cnt) != 0) {
		printf("An error occured: %d\n", ret);
	} else {
		printf("test_cnt = %d\n", test_cnt);
	}

	// Verifying the image and label data of the specified number of examples
	//printExamples(&test_set, 1);

	// Verifying the convolutional layer
	double data[INSIZE][INSIZE];

 	//data[0][0] = 0.0f; 
	for (i = 0; i < INSIZE; i++) {
		for (int j = 0; j < INSIZE; j++){
			data[i][j] = INPUT; //1.0f;
      		//data[i][j+1] =  data[i][j] + 1.0f; 
    	}
	}	

	forward_pass(data, true);
	#if !SIMULATION
	copy_trained_parameters();
	#endif 

	// Performing forward pass
	unsigned int error = 0;
	unsigned int max = 0;
	float res[10];

	#if SIMULATION	
	for (i = 0; i < 1; i++) { //test_cnt
		time_taken += forward_pass(test_set[i].data, true);
	#else
	for (i = 0; i < test_cnt; i++) { //test_cnt
		time_taken += forward_pass(test_set[i].data, false);
	#endif 
		cudaMemcpy(res, fc_layer.output, sizeof(float) * NUM_CLASSES, cudaMemcpyDeviceToHost);

		for (int j = 0; j < NUM_CLASSES; j++) {
			if (res[max] < res[j])
				max = j;
		}

		if (max != test_set[i].label)
			error++;
	}

	printf("Error Rate = %f%% (%d out of 10000)\n", double(error) / double(test_cnt) * 100.0, error);
	printf("Accuracy = %.3f%% (%d out of 10000)\n", 100.0 - double(error) / double(test_cnt) * 100.0, test_cnt - error);
	printf("Execution time = %f (ms) \n\n", time_taken);

	free(test_set);
	return 0;
}

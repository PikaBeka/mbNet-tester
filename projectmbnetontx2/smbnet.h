#include <stdio.h>

typedef struct layer{
	int layer_type; //0: conv 1:subsample 2:fully connected  
	int input_size; 
	int filter_nr; 
	int kernel_size; 
	int stride; 
	void (*function_name)(...); // if needed 
}layer; 

typedef struct network{
	int network_nr; //0: 3-layer mbNet (now) (later 1: 5-layer LeNet-1, 2: 16-layer darknet)
	int nr_layers; 
	int batch; 
	layer *layers; 
}network; 


#define ALL_LAYERS 1

// For configurations 
//1. Composite: 	CONV_COMPOSITE = 1, DIRECT = 1/0
//2. Direct: 		CONV_COMPOSITE = 0, DIRECT = 1
//3. GEMM: 		CONV_COMPOSITE = 0, DIRECT = 0 
 
#define SIMULATION 0
#define DIRECT 1

#define CONV_COMPOSITE 0
#define SS_COMPOSITE 1
#define FC_COMPOSITE 1

#define CPU_GEMM 0 //0: GPU 1: CPU 
#define GEMM_GLOBAL 0 //1: Global 0:cublas
#define CONV_SHARED 1

#define NUM_CLASSES 10

#define INPUT 1.0f //-1.0f // 
#define WEIGHT 1.0f // -1.0f // 
#define BIAS 1.0f //-1.0f // 
#define CONV_POST_ACT 1.0f // 1.0f //  
#define CONV2_POST_ACT 1.0f // 1.0f //  
#define SS_POST_ACT 1.0f // 0.0f // 
#define SS2_POST_ACT 1.0f // 0.0f // 
#define FC_POST_ACT 1.0f // (1/(1+2.71828)) // 

// Convolution 1 
#define INCH 1
#define INSIZE 28
#define FILTER_SIZE 5
#define STRIDE 1
#define PAD	0
#define CHANNEL 6 //4
#define CONV_OUTPUT_SIZE ((INSIZE - FILTER_SIZE)/STRIDE + 1 + 2*PAD) //24

#define N1 CHANNEL*CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE
#define K1 64

// for GEMM 
#define N11 CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE
#define K11 64


#define SHBS 8 //8:37.502ms  12: 42.389ms //6: 74.306ms  
#define BS (SHBS-FILTER_SIZE+1) // 8
#define GRID (CONV_OUTPUT_SIZE/BS) // 3 
#define CONV_NB (GRID*GRID) //9 
#define CONV_TPB MAX(CHANNEL*BS*BS, CHANNEL*FILTER_SIZE*FILTER_SIZE) //at least bigger than 150 (25x6)

// Pooling 1 (Subsampling) 
#define SS_SIZE 4
#define SS_STRIDE 4
#define SS_CHANNELS 1
#define SS_OUTPUT_SIZE ((CONV_OUTPUT_SIZE - SS_SIZE)/SS_STRIDE + 1) //6

#define N2 (CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE)
#define K2 64 //8

// Fully Connection 1 
#define FC_SHARED 1
#if FC_SHARED 
#define N3 (NUM_CLASSES*CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE)
#define K3 (CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE)
#else
#define N3 NUM_CLASSES*CHANNEL*SS_OUTPUT_SIZE*SS_OUTPUT_SIZE
#define K3 NUM_CLASSES
//#define N3 64
//#define K3 64
#endif 

#define CONV2 0

//NOT used for the base mbNet 
#define LENET5_C2 0
#define LENET5_SS2 0
#define LENET5_FC 0

/*
#define SHBS2 12//12 // 8 
#define FILTER_SIZE2 5 //1
#define BS2 (SHBS2-FILTER_SIZE2+1) // 8 

#define INNR CHANNEL //6
#define CHANNEL2 12
#define CONV2_OUTPUT_SIZE ((SS_OUTPUT_SIZE - FILTER_SIZE2)/STRIDE + 1)
#define GRID2 (CONV2_OUTPUT_SIZE/BS2) // 3
#define CONV_NB2 (GRID2*GRID2) //3x3
#define CONV_TPB2 MAX(CHANNEL2*BS2*BS2, CHANNEL2*FILTER_SIZE2*FILTER_SIZE2) //at least bigger than 150 (25x6)


#define SS2_SIZE 2//4
#define SS2_STRIDE 2//4
#define SS2_CHANNELS 1
#define SS2_OUTPUT_SIZE ((CONV2_OUTPUT_SIZE - SS2_SIZE)/SS2_STRIDE + 1) //6

#define N22 (CHANNEL2*SS2_OUTPUT_SIZE*SS2_OUTPUT_SIZE)
#define K22 8 //64

#define N33 (NUM_CLASSES*CHANNEL2*SS2_OUTPUT_SIZE*SS2_OUTPUT_SIZE)
#define K33 (CHANNEL2*SS2_OUTPUT_SIZE*SS2_OUTPUT_SIZE)


__global__ void kernel_conv2_filter(float input[INNR][SS_OUTPUT_SIZE][SS_OUTPUT_SIZE],
									float pre_output[CHANNEL2][CONV2_OUTPUT_SIZE][CONV2_OUTPUT_SIZE],
                                  	float weight[CHANNEL2][FILTER_SIZE2][FILTER_SIZE2], 
                                    float bias[CHANNEL2],
                                    float output[CHANNEL2][CONV2_OUTPUT_SIZE][CONV2_OUTPUT_SIZE]); 


#if 1 //SS2_COMPOSITE
__global__ void kernel_ss2_composite(float input[CHANNEL2][CONV2_OUTPUT_SIZE][CONV2_OUTPUT_SIZE],
#else
__global__ void kernel_ss2_filter(float input[CHANNEL2][CONV2_OUTPUT_SIZE][CONV2_OUTPUT_SIZE],
#endif  
								float pre_output[CHANNEL2][SS2_OUTPUT_SIZE][SS2_OUTPUT_SIZE], 
								float weight[SS2_CHANNELS][SS2_SIZE][SS2_SIZE],
								float bias[SS2_CHANNELS], float output[CHANNEL2][SS2_OUTPUT_SIZE][SS2_OUTPUT_SIZE]);


#if 1 //FC_COMPOSITE
__global__ void kernel_fc1_lenet5_composite(float input[CHANNEL2][SS2_OUTPUT_SIZE][SS2_OUTPUT_SIZE], 
#else 
__global__ void kernel_fc1(float input[CHANNEL2][SS2_OUTPUT_SIZE][SS2_OUTPUT_SIZE], 
#endif 
							float pre_output[NUM_CLASSES], 
                            float weight[NUM_CLASSES][CHANNEL2][SS2_OUTPUT_SIZE][SS2_OUTPUT_SIZE],
							float bias[NUM_CLASSES], float output[NUM_CLASSES]);


*/

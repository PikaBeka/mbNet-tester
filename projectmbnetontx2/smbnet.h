#include <stdio.h>

typedef struct layer
{
	int layer_type; // 0: conv 1:subsample 2:fully connected
	int input_size;
	int filter_nr;
	int kernel_size;
	int stride;
	void (*function_name)(...); // if needed
} layer;

typedef struct network
{
	int network_nr; // 0: 3-layer mbNet (now) (later 1: 5-layer LeNet-1, 2: 16-layer darknet)
	int nr_layers;
	int batch;
	layer *layers;
} network;

#define ALL_LAYERS 1

// For configurations
// 1. Composite: 	CONV_COMPOSITE = 1, DIRECT = 1/0
// 2. Direct: 		CONV_COMPOSITE = 0, DIRECT = 1
// 3. GEMM: 		CONV_COMPOSITE = 0, DIRECT = 0

#define SIMULATION 0
#define DIRECT 1

#define CONV_COMPOSITE 0
#define SS_COMPOSITE 1
#define FC_COMPOSITE 1

#define CPU_GEMM 0	  // 0: GPU 1: CPU
#define GEMM_GLOBAL 0 // 1: Global 0:cublas
#define CONV_SHARED 1

#define NUM_CLASSES 10

#define INPUT 1.0f			//-1.0f //
#define WEIGHT 1.0f			// -1.0f //
#define BIAS 1.0f			//-1.0f //
#define CONV_POST_ACT 1.0f	// 1.0f //
#define CONV2_POST_ACT 1.0f // 1.0f //
#define SS_POST_ACT 1.0f	// 0.0f //
#define SS2_POST_ACT 1.0f	// 0.0f //
#define FC_POST_ACT 1.0f	// (1/(1+2.71828)) //

// Convolution 1
#define INCH 1
#define INSIZE 28
#define FILTER_SIZE 5
#define STRIDE 1
#define PAD 0
#define CHANNEL 6														 // 4
#define CONV_OUTPUT_SIZE ((INSIZE - FILTER_SIZE) / STRIDE + 1 + 2 * PAD) // 24

#define N1 CHANNEL *CONV_OUTPUT_SIZE *CONV_OUTPUT_SIZE
#define K1 64

// for GEMM
#define N11 CONV_OUTPUT_SIZE *CONV_OUTPUT_SIZE
#define K11 64

#define SHBS 8															 // 8:37.502ms  12: 42.389ms //6: 74.306ms
#define BS (SHBS - FILTER_SIZE + 1)										 // 8
#define GRID (CONV_OUTPUT_SIZE / BS)									 // 3
#define CONV_NB (GRID * GRID)											 // 9
#define CONV_TPB MAX(CHANNEL *BS *BS, CHANNEL *FILTER_SIZE *FILTER_SIZE) // at least bigger than 150 (25x6)

// Pooling 1 (Subsampling)
#define SS_SIZE 4
#define SS_STRIDE 4
#define SS_CHANNELS 1
#define SS_OUTPUT_SIZE ((CONV_OUTPUT_SIZE - SS_SIZE) / SS_STRIDE + 1) // 6

#define N2 (CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE)
#define K2 64 // 8

// Fully Connection 1
#define FC_SHARED 1
#if FC_SHARED
#define N3 (NUM_CLASSES * CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE)
#define K3 (CHANNEL * SS_OUTPUT_SIZE * SS_OUTPUT_SIZE)
#else
#define N3 NUM_CLASSES *CHANNEL *SS_OUTPUT_SIZE *SS_OUTPUT_SIZE
#define K3 NUM_CLASSES
//#define N3 64
//#define K3 64
#endif

#define CONV2 0

// NOT used for the base mbNet
#define LENET5_C2 0
#define LENET5_SS2 0
#define LENET5_FC 0
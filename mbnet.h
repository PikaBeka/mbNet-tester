#define N 1 // number of images in batch
#define C 16
#define HW 16

#define K 32
#define RS 5 // kernel height and width

#define PQ (HW - RS + 1) // output height and width (146)

#define TILE_S 8
#define LIM (TILE_S - RS + 1) // 4

#define ARRAY_NAIVE 1
#define ARRAY_TILING 0
#define DIRECT 0
#define GEMM_GLOBAL 0

#define CONV_SHARED 0
#define GRID ((PQ + LIM - 1) / LIM)                     // (37)
#define CONV_NB (GRID * GRID)                           // 1369
#define CONV_TPB MIN(1024, MAX(K *LIM *LIM, K *RS *RS)) // threads per block (150)

#define UNROLL_TPB 64
#define UNROLL_NB (PQ * PQ * C)
#define STRIDE 1

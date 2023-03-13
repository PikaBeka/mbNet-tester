#define N 10000 // number of images in batch
#define BATCH 10
#define C 1
#define HW 400
#define K 6
#define RS 5             // kernel height and width
#define PQ (HW - RS + 1) // output height and width (146)
#define TILE_S 8
#define LIM (TILE_S - RS + 1) // 4
#define STRIDE 1

#define ARRAY_NAIVE 0

#define ARRAY_TILING 0

#define DIRECT 0
#define CONV_SHARED 0
#define GRID ((PQ + LIM - 1) / LIM)                     // (37)
#define CONV_NB (GRID * GRID)                           // 1369
#define CONV_TPB MIN(1024, MAX(K *LIM *LIM, K *RS *RS)) // threads per block (150)

#define CUDNN 1

#define GEMM_GLOBAL 0
#define UNROLL_TPB MIN(1024, K *RS *RS)
#define UNROLL_NB ((PQ * PQ * C + UNROLL_TPB) / UNROLL_TPB)

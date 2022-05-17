#ifndef __DATE__
#define __DATE__
#endif

#ifndef __TIME__
#define __TIME__
#endif

#ifndef __FUNCTION__
#define __FUNCTION__
#endif

#ifndef __LINE__
#define __LINE__ 0
#endif

#ifndef __FILE__
#define __FILE__
#endif

void check_error(cudaError_t status);
void check_error_extended(cudaError_t status, const char *file, int line, const char *date_time);
void cublas_check_error_extended(cublasStatus_t status, const char *file, int line, const char *date_time);
#define CHECK_CUDA(X) check_error_extended(X, __FILE__ " : " __FUNCTION__, __LINE__,  __DATE__ " - " __TIME__ );
#define CHECK_CUBLAS(X) cublas_check_error_extended(X, __FILE__ " : " __FUNCTION__, __LINE__,  __DATE__ " - " __TIME__ );

    cublasHandle_t blas_handle();
    void free_pinned_memory();
    void pre_allocate_pinned_memory(size_t size);
    float *cuda_make_array_pinned_preallocated(float *x, size_t n);
    float *cuda_make_array_pinned(float *x, size_t n);
    float *cuda_make_array(float *x, size_t n);
    void **cuda_make_array_pointers(void **x, size_t n);
    int *cuda_make_int_array(size_t n);
	int *cuda_make_int_array_new_api(int *x, size_t n);
    void cuda_push_array(float *x_gpu, float *x, size_t n);
    //LIB_API void cuda_pull_array(float *x_gpu, float *x, size_t n);
    //LIB_API void cuda_set_device(int n);
    int cuda_get_device();
    void cuda_free_host(float *x_cpu);
    void cuda_free(float *x_gpu);
    void cuda_random(float *x_gpu, size_t n);
    float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
    dim3 cuda_gridsize(size_t n);
    cudaStream_t get_cuda_stream();
    //cudaStream_t get_cuda_memcpy_stream();
    int get_number_of_blocks(int array_size, int block_size);
    int get_gpu_compute_capability(int i, char *device_name);
    void show_cuda_cudnn_info();

    cudaStream_t switch_stream(int i);
    void wait_stream(int i);
    void reset_wait_stream_events();


int cuda_debug_sync = 0;

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    CHECK_CUDA(status);
    return n;
}

void check_error(cudaError_t status)
{
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("\n CUDA Error: %s\n", s);
        snprintf(buffer, 256, "CUDA Error: %s", s);
#ifdef WIN32
        getchar();
#endif
        //error(buffer, DARKNET_LOC); //jurn 
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status2);
        char buffer[256];
        printf("\n CUDA Error Prev: %s\n", s);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
#ifdef WIN32
        getchar();
#endif
        //error(buffer, DARKNET_LOC); //jurn 
    }
}

void check_error_extended(cudaError_t status, const char *file, int line, const char *date_time)
{
    if (status != cudaSuccess) {
        printf("CUDA status Error: file: %s() : line: %d : build time: %s \n", file, line, date_time);
        check_error(status);
    }
#if defined(DEBUG) || defined(CUDA_DEBUG)
    cuda_debug_sync = 1;
#endif
    if (cuda_debug_sync) {
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            printf("CUDA status = cudaDeviceSynchronize() Error: file: %s() : line: %d : build time: %s \n", file, line, date_time);
    }
    check_error(status);
}



void cublas_check_error(cublasStatus_t status)
{
#if defined(DEBUG) || defined(CUDA_DEBUG)
    cudaDeviceSynchronize();
#endif
    if (cuda_debug_sync) {
        cudaDeviceSynchronize();
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS Error\n");
    }
}

void cublas_check_error_extended(cublasStatus_t status, const char *file, int line, const char *date_time)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("\n cuBLAS status Error in: file: %s() : line: %d : build time: %s \n", file, line, date_time);
    }
#if defined(DEBUG) || defined(CUDA_DEBUG)
    cuda_debug_sync = 1;
#endif
    if (cuda_debug_sync) {
        cudaError_t status = cudaDeviceSynchronize();
      if (status != 0) //CUDA_SUCCESS = 0
          printf("\n cudaError_t status = cudaDeviceSynchronize() Error in: file: %s() : line: %d : build time: %s \n", file, line, date_time);
    }
    cublas_check_error(status);
}

static cudaStream_t streamsArray[16];    // cudaStreamSynchronize( get_cuda_stream() );
static int streamInit[16] = { 0 };

cudaStream_t get_cuda_stream() {
    int i = cuda_get_device();
    if (!streamInit[i]) {
        printf("Create CUDA-stream - %d \n", i);
#ifdef CUDNN
        cudaError_t status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamNonBlocking);
#else
        cudaError_t status = cudaStreamCreate(&streamsArray[i]);
#endif
        if (status != cudaSuccess) {
            printf(" cudaStreamCreate error: %d \n", status);
            const char *s = cudaGetErrorString(status);
            printf("CUDA Error: %s\n", s);
            status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamNonBlocking);    // cudaStreamDefault
            CHECK_CUDA(status);
        }
        streamInit[i] = 1;
    }
    return streamsArray[i];
}


static int blasInit[16] = { 0 };
static cublasHandle_t blasHandle[16];

cublasHandle_t blas_handle()
{
    int i = cuda_get_device();
    if (!blasInit[i]) {
        CHECK_CUBLAS(cublasCreate(&blasHandle[i]));
        cublasStatus_t status = cublasSetStream(blasHandle[i], get_cuda_stream());
        CHECK_CUBLAS(status);
        blasInit[i] = 1;
    }
    return blasHandle[i];
}
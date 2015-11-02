#include <sys/types.h>
#include <sys/mman.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda_runtime_api.h>


#define GCHECK(ans) { gAssert((ans), __FILE__, __LINE__); }
inline void gAssert(cudaError_t code, const char *file, int line)
{
    char s[4096] = {0};
    if (code != cudaSuccess) 
    {
        sprintf(s, "GPUError: %s %s %d\n", cudaGetErrorString(code), file, line);
        mexErrMsgIdAndTxt("MATLAB:gpup2p", s);
    }
}
typedef struct gpup2p
{
    cudaIpcMemHandle_t memHandle;
    int len;
} gpup2p;
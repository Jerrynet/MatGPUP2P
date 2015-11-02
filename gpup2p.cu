#include "gpup2p.h"

static char filePath[4096] = {0};
static int nProc = 1;
static int fd = -1;
static gpup2p *gp;
static mxGPUArray *myGPUArray;
static float * flower;

__global__ void addGop(float const *d1, float *d2, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    d2[index] += d1[index];
}

void mexFunction( int nlhs,       mxArray *plhs[], 
                  int nrhs, const mxArray *prhs[]  )
{
    const int op = *(int*)mxGetData(prhs[0]);
    if (op==-1 && nrhs==3)
    {
        int id1 = *(int *)mxGetData(prhs[1])-1;
        int id2 = *(int *)mxGetData(prhs[2])-1;
        int accessible = 0;
        GCHECK(cudaDeviceCanAccessPeer( &accessible, id1, id2 ));
        plhs[0] = mxCreateDoubleScalar((double) accessible);
        return;
    }
    else if (op==0 && nrhs==3)
    {
        mxGetString(prhs[1], filePath, mxGetN(prhs[1])+1);
        nProc = *(int *)mxGetData(prhs[2]);
        if (fd != -1) 
        {
            close(fd);
            fd = -1;
        }

        FILE *fw;
        fw = fopen(filePath, "w");
        if(fw != NULL)
        {
            char *zeros;
            zeros = (char *)malloc(sizeof(gpup2p)*nProc);
            memset(zeros, 0, sizeof(gpup2p)*nProc);
            fwrite(zeros, sizeof(char), sizeof(gpup2p)*nProc, fw);
            fclose(fw);
            free(zeros);
        }

        fd = open(filePath, O_CREAT | O_EXCL, 0);
        if (fd == -1 && errno == EEXIST )
        {
            fd = open(filePath, O_RDWR, 0);
            if (fd == -1) mexErrMsgIdAndTxt("MATLAB:gpup2p", "open files error.");
        }
        else if (fd != -1)
        {
            close(fd);
            fd = -1;
            fd = open(filePath, O_RDWR, 0);
            if (fd == -1) mexErrMsgIdAndTxt("MATLAB:gpup2p", "open files error.");
        }
        else
        {
            mexErrMsgIdAndTxt("MATLAB:gpup2p", "create files error.");
            return;
        }
        gp = (gpup2p*)mmap(NULL, sizeof(gpup2p)*nProc, PROT_READ|PROT_WRITE, MAP_FILE|MAP_SHARED, fd, 0);
        if (gp == MAP_FAILED) mexErrMsgIdAndTxt("MATLAB:gpup2p", "mmap failed.");
        mxInitGPU();
    }
    else if (op==1 && nrhs==3)
    {
        int labindex = *(int *)mxGetData(prhs[1]) -1;

        myGPUArray = mxGPUCopyFromMxArray(prhs[2]);
        gp[labindex].len  = (int)mxGPUGetNumberOfElements(myGPUArray);
        GCHECK(cudaMalloc((void**) &flower, gp[labindex].len*sizeof(float)));
        GCHECK(cudaMemcpy(flower, (float *)mxGPUGetData(myGPUArray), gp[labindex].len*sizeof(float), cudaMemcpyDeviceToDevice));
        
        //mxGPUArray const *ohmy = mxGPUCreateFromMxArray(prhs[2]);
        //flower = (float *)mxGPUGetDataReadOnly(ohmy);
        //mxGPUDestroyGPUArray(ohmy);
        
        GCHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &(gp[labindex].memHandle), (void *) flower));
    }
    else if (op==2 && nrhs==2)
    {
        int labindex = *(int *)mxGetData(prhs[1]) -1;

        int const threadsPerBlock = 256;
        int blocksPerGrid = (gp[labindex].len + threadsPerBlock - 1) / threadsPerBlock;
        cudaError_t err;

        float *tmptr;
        float *finals = (float *)mxGPUGetData(myGPUArray);

        //float A=0.0, B=0.0;
        for(int i=0;i<nProc;i++){
            if (labindex==i) continue;
            GCHECK(cudaIpcOpenMemHandle((void **) &tmptr, gp[i].memHandle, cudaIpcMemLazyEnablePeerAccess));
            // GCHECK(cudaMemcpy(&A, tmptr, sizeof(float), cudaMemcpyDeviceToHost));
            // GCHECK(cudaMemcpy(&B, finals, sizeof(float), cudaMemcpyDeviceToHost));
            // printf(" (%.0f + %.0f)\n", A, B);
            addGop<<<blocksPerGrid, threadsPerBlock>>>(tmptr, finals, gp[i].len);
            //cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess) mexErrMsgIdAndTxt("MATLAB:gpup2p", "kernel execution failed.");
            GCHECK(cudaIpcCloseMemHandle(tmptr));
        }
        
    }
    else if (op==3 && nrhs==2)
    {
        cudaFree(flower);
        int labindex = *(int *)mxGetData(prhs[1]) -1;
        plhs[0] = mxGPUCreateMxArrayOnGPU(myGPUArray);
        mxGPUDestroyGPUArray(myGPUArray);
    }
    else if (op==4 && nrhs==1) //destroy
    {
        munmap(gp, sizeof(gpup2p)*nProc);
        close(fd);
        nProc = 1;
        memset(filePath, 0, 4096);
        fd = -1;
    }
    else
    {
        mexErrMsgIdAndTxt("MATLAB:gpup2p", "Input error.");
    }

    
}
# MatGPUP2P
Lightweight GPU Peer to Peer Operation for Matlab

## Functionality
Designed for *spmd* function, and replace the built-in *gop()* for fast 
array operation without copying it to your host memory.

## Limitation
- Only works on single computer with nVIDIA graphics cards.
- Only works on Linux version of MATLAB
- Only supports 'single' datatype on MATLAB (currently)
- Many graphics cards do not support P2P access, you can check it with this command: `gpuop('check', {GPUID1}, {GPUID2})`
# CUDA-Cpp_paralelized_Ground-State-DMRG
Ground-State MPS-DMRG implementation using GPU paralelization on CUDA-C++

## Version 1
There's a first version, Lanczos-Method-only, in which everything's managed locally (CPU), except for Lanczos routines, which are relayed to GPU every time needed, results being thrown back thereafter, returning to main thread excecution.

## Version 2
The more optimized version, Everything_on_GPU, as its name indicates, performs the whole algorithm externally (GPU), further improving efficiency (lowering calculation times) as it eliminates the need for constant data transfering between CPU and GPU (through relatively low rate PCI bus interfaces), as everything is kept on the graphics card's storage. 

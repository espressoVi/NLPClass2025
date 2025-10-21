# GPU computation demo

This folder contains code (```C```) demonstrating massively parallel computation
with GPUs.


## Usage

> [!WARNING]
> A CUDA capable GPU is required for execution.

* Compile the _CPU_ code with standard ```gcc```, and run. The commands are as
follows:

```
$ gcc -o cpu cpu.c
$ ./cpu
```

* Compile the _GPU_ version with the following:

```
nvcc -ccbin=/usr/bin/g++-14 -O3 -o gpu gpu.cu
```

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np

# Define the kernel function
mod = SourceModule("""
    /* Update the temperature values using five-point stencil */
    __global__ void evolve_kernel(double *currdata, double *prevdata, double a, double dt, double dx2, double dy2,
                                    int nx, int ny
                           )
    {
    
        /* Determine the temperature field at next time step
         * As we have fixed boundary conditions, the outermost gridpoints
         * are not updated. */
        int ind, ip, im, jp, jm;
    
        // CUDA threads are arranged in column major order; thus j index from x, i from y
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;        
    
        if (i > 0 && j > 0 && i < nx+1 && j < ny+1) {
            ind = i * (ny + 2) + j;
            ip = (i + 1) * (ny + 2) + j;
            im = (i - 1) * (ny + 2) + j;
        jp = i * (ny + 2) + j + 1;
        jm = i * (ny + 2) + j - 1;
            currdata[ind] = prevdata[ind] + a * dt *
              ((prevdata[ip] -2.0 * prevdata[ind] + prevdata[im]) / dx2 +
              (prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm]) / dy2);
    
        }
    
    }
    
""")


# Get the kernel function
def get_cuda_function():
    return mod.get_function("evolve_kernel")

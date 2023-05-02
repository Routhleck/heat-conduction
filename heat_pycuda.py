import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np

# Define the kernel function
mod = SourceModule("""
    /* Update the temperature values using five-point stencil */
    __global__ void evolve_kernel(double* u, double* u_previous,
                                  double a, double dt, double dx2, double dy2,
                                  int nx, int ny) {
        int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
        if (i >= nx - 1 || j >= ny - 1) return;
        u[i * ny + j] = u_previous[i * ny + j] + a * dt * (
                        (u_previous[(i + 1) * ny + j] - 2 * u_previous[i * ny + j] +
                         u_previous[(i - 1) * ny + j]) / dx2 +
                        (u_previous[i * ny + j + 1] - 2 * u_previous[i * ny + j] +
                         u_previous[i * ny + j - 1]) / dy2);
        u_previous[i * ny + j] = u[i * ny + j];
    }
""")


# Get the kernel function
def get_cuda_function():
    return mod.get_function("evolve_kernel")

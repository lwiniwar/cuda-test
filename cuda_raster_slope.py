# from osgeo import gdal
import time

import numba
from numba import cuda
import numpy as np
import math



@cuda.jit
def get_slope_cuda(arrx, arry, Sx, SxSx, out):
    cx, cy = cuda.grid(2)
    n = len(arrx)
    Sy = 0
    # Sx = 0
    Sxy = 0
    Sx2 = 0
    for itemx, itemy in zip(arrx, arry[cx, cy]):
        Sxy += itemx * itemy
        Sx2 += itemx * itemx
        # Sx += itemx
        Sy += itemy
    out[cx, cy]= (n*Sxy - Sx * Sy) / (n*Sx2 - SxSx)


if __name__ == '__main__':
    np.random.seed(42)
    testdata = np.random.randint(1000, 5000, (8000, 30000, 10))#.astype(float)
    x = np.arange(6)
    Sx = np.sum(x)
    SxSx = Sx * Sx
    # an_array = np.zeros(testdata.shape[:-1], dtype=float)

    print("Data ready.")
    # threadsperblock = (2, 2)
    # threadsperblock = (16, 16)
    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(testdata.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(testdata.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    t0 = time.time()
    stream = cuda.stream()

    x = cuda.to_device(x)
    Sx = cuda.to_device(Sx)
    SxSx = cuda.to_device(SxSx)
    testdata = cuda.to_device(np.ascontiguousarray(testdata)) # transfer to GPU manually, so it won't be transferred back
    an_array = cuda.device_array(testdata.shape[:-1], dtype=float)  # create on GPU directly, no need for copying

    get_slope_cuda[blockspergrid, threadsperblock, stream](x, testdata, Sx, SxSx, an_array)
    print(an_array.copy_to_host())
    t1 = time.time()
    print(f"Took {(t1-t0)}s. (Array dim: {testdata.shape}")

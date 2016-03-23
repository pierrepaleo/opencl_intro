import numpy as np
from oclutils import Ocl
import matplotlib.pyplot as plt


def myhist256(img):
    res = np.clip(img.astype(np.int32), 0, 255)
    h = np.histogram(res, bins=256, range=(0, 255))
    return h[0]


if __name__ == "__main__":

    from scipy.misc import lena
    img = lena().astype(np.int32) # !

    Nr, Nc = img.shape

    # Initialize device
    ocl = Ocl(profile=True)
    print("running on %s" % ocl.devicename)

    # Build Program
    program = ocl.compile_file("opencl/histogram.cl")

    # Device memory
    d_input = ocl.to_device(img, flags="r")
    d_output = ocl.create_buffer_zeros(256, np.int32)

    # Execute
    wg = None
    shape = (Nc, Nr)
    kern = ocl.call(program.histogram256, shape, wg, d_input, d_output, Nr, Nc)

    # Retrieve result
    res = ocl.fetch(d_output)

    print("Max err: %e" % np.max(np.abs(res - myhist256(img))))
    # Show
    plt.figure()
    plt.plot(res);
    plt.show()

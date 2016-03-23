import numpy as np
from oclutils import Ocl
import matplotlib.pyplot as plt


def mybin2(img):
    res = img.astype(np.float32)
    Nr, Nc = img.shape
    Nr2, Nc2 = Nr/2, Nc/2
    res = res.reshape((Nr2, 2, Nc2, 2)).sum(axis=1).sum(axis=-1)
    return res*0.25


if __name__ == "__main__":

    from scipy.misc import lena
    img = lena().astype(np.float32)
    Nr, Nc = img.shape
    Nr2, Nc2 = Nr/2, Nc/2

    # Initialize device
    ocl = Ocl(profile=True)
    print("running on %s" % ocl.devicename)

    # Build Program
    program = ocl.compile_file("opencl/binning.cl")

    # Device memory
    d_input = ocl.to_device(img, flags="r")
    d_output = ocl.create_buffer_zeros((Nr2, Nc2), np.float32)

    # Execute
    wg = None # unspecified workgroup size
    shape = (Nc2, Nr2)
    kern = ocl.call(program.binning2, shape, wg, d_input, d_output, Nr2, Nc2)

    # Retrieve result
    res = ocl.fetch(d_output)

    print("Max err = %e" % np.max(np.abs(res - mybin2(img))))
    # Show
    plt.figure()
    plt.imshow(res, cmap="gray", interpolation="nearest"); plt.colorbar()
    plt.show()




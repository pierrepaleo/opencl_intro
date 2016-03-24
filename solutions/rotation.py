import numpy as np
from oclutils import Ocl
import matplotlib.pyplot as plt




if __name__ == "__main__":

    from scipy.misc import lena
    img = lena().astype(np.float32)
    Nr, Nc = img.shape

    # Initialize device
    ocl = Ocl(profile=True)
    print("running on %s" % ocl.devicename)

    # Build Program
    program = ocl.compile_file("opencl/rotation.cl")

    # Device memory
    d_input = ocl.to_device(img, flags="r")
    d_output = ocl.create_buffer_zeros((Nr, Nc), np.float32)

    # Execute
    theta = np.float32(0.785)
    wg = None # unspecified workgroup size
    shape = (Nc, Nr)
    kern = ocl.call(program.rotation, shape, wg, d_input, d_output, Nr, Nc, 255, 255, theta)

    # Retrieve result
    res = ocl.fetch(d_output)

    # Show
    plt.figure()
    plt.imshow(res, cmap="gray", interpolation="nearest"); plt.colorbar()
    plt.show()


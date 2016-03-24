#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from oclutils import Ocl
import matplotlib.pyplot as plt
try:
    from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    __has_ndimage = True
except ImportError:
    __has_ndimage = False


def comp_kern_scipy(sigma, truncate=4):
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return np.ascontiguousarray(weights)





if __name__ == "__main__":

    from scipy.misc import lena
    img = lena().astype(np.float32)
    Nr, Nc = img.shape
    sigma = 1.6
    gaussian = comp_kern_scipy(sigma)
    ksize = gaussian.shape[0]

    # Initialize device
    ocl = Ocl(profile=True)
    print("running on %s" % ocl.devicename)

    # Build Program
    program = ocl.compile_file("opencl/separable_nonseparable.cl")

    # Device memory
    d_input = ocl.to_device(img, flags="r")
    d_gaussian = ocl.to_device(gaussian, flags="r")
    d_output = ocl.create_buffer_zeros((Nr, Nc), np.float32)
    d_tmp = ocl.create_buffer_zeros((Nr, Nc), np.float32)

    # Separable convolution
    # -----------------------
    wg = None
    shape = (Nc, Nr)
    k1 = ocl.call(program.horizontal_convolution, shape, wg, d_input, d_tmp, d_gaussian, ksize, Nc, Nr)
    k1.wait()
    print("Horizontal convolution took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start)))
    k2 = ocl.call(program.vertical_convolution, shape, wg, d_tmp, d_output, d_gaussian, ksize, Nc, Nr)
    k2.wait()
    print("Vertical convolution took %.3fms" % (1e-6 * (k2.profile.end - k2.profile.start)))

    # Retrieve result
    res = ocl.fetch(d_output)

    if __has_ndimage:
        print("Max err : %e" % np.max(np.abs(res - scipy_gaussian_filter(img, sigma))))


    # Non-separable convolution
    # ------------------------------
    gaussian2 = np.outer(gaussian, gaussian)
    d_gaussian = ocl.to_device(gaussian2, flags="r")

    wg = None
    shape = (Nc, Nr)
    k3 = ocl.call(program.nonseparable_convolution, shape, wg, d_input, d_output, d_gaussian, ksize, Nc, Nr)
    k3.wait()
    print("Nonseparable convolution took %.3fms" % (1e-6 * (k3.profile.end - k3.profile.start)))

    # Retrieve result
    res = ocl.fetch(d_output)

    if __has_ndimage:
        print("Max err : %e" % np.max(np.abs(res - scipy_gaussian_filter(img, sigma))))



    # Show
    # plt.figure()
    # plt.imshow(res, cmap="gray", interpolation="nearest");
    # plt.show()


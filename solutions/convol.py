#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy, scipy.misc, scipy.ndimage
import math
import time
from oclutils import Ocl


def scipy_gaussianfilter(img, sigma):
    return scipy.ndimage.gaussian_filter(img, sigma)#, mode="reflect")


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



class Gpuconvol:
    """
    Helper class for 3D convolution on GPU.
    """


    def __init__(self, shape, device=None, program_path=None):
        """
        Initialize the Gpu Convolution : context, temporary/output device arrays
        and program.
        """

        # Create the GPU context
        self.ocl = Ocl(device=device)

        # Pre-allocate the arrays : input, output and tmp
        nbytes = np.prod(shape)*4
        self.d_input = self.ocl.create_buffer(shape, np.float32, flags="r")
        self.d_output = self.ocl.create_buffer(shape, np.float32, flags="w")
        self.d_tmp = self.ocl.create_buffer(shape, np.float32)

        # Compile the OpenCL kernels
        if program_path is None:
            program_path = "opencl/convolution.cl"
        self.program = self.ocl.compile_file(program_path)


        # Prepare the grid/block size
        self.ndim = len(shape)
        if self.ndim == 3:
            im_z, im_h, im_w = shape
            self.shape = (im_w, im_h, im_z)
        else: #elif self.ndim == 2:
            im_h, im_w = shape
            self.shape = (im_w, im_h)
        self.wg = None

        self.wg = (4, 4, 4) if (self.ndim == 3) else (4, 4, 1) # For CC <= 2.0, no more than 512 threads/block (1024 otherwise)
        self.grid = self.ocl.calc_size(self.shape, self.wg)


    def gaussian_filter(self, image, sigma):
        # Compute the gaussian filter
        gaussian = comp_kern_scipy(sigma)

        # Transfer the image and the gaussian kernel
        d_input = self.ocl.to_device(image, destbuf=self.d_input, flags="r")
        d_gaussian = self.ocl.to_device(gaussian, flags="r")

        # Check that the size is valid
        if self.ndim == 3: # 3D
            if (image.shape[0] != self.shape[2]) or (image.shape[1] != self.shape[1]) or (image.shape[2] != self.shape[0]):
                raise ValueError('gaussian_filter(): invalid volume size: expected (%d, %d, %d), got (%d, %d, %d)' % (self.shape[2], self.shape[1], self.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        else: #2D
            if (image.shape[0] != self.shape[1]) or (image.shape[1] != self.shape[0]):
                raise ValueError('gaussian_filter(): invalid volume size: expected (%d, %d), got (%d, %d)' % (self.shape[1], self.shape[0], image.shape[0], image.shape[1]))

        # Execute
        ksize = gaussian.shape[0]
        if self.ndim == 3: # 3D
            im_z, im_h, im_w = self.shape
        else: #2D
            im_h, im_w = self.shape
            im_z = 1
        k1 = self.ocl.call(self.program.horizontal_convolution, self.grid, self.wg, d_input, self.d_output, d_gaussian, ksize, im_w, im_h, im_z)
        k2 = self.ocl.call(self.program.vertical_convolution, self.grid, self.wg, self.d_output, self.d_tmp, d_gaussian, ksize, im_w, im_h, im_z)
        if self.ndim == 3:
            k3 = self.ocl.call(self.program.depth_convolution, self.grid, self.wg, self.d_tmp, self.d_output, d_gaussian, ksize, im_w, im_h, im_z)

        # Free the memory for gaussian kernel
        self.ocl.release_buffer(d_gaussian)

    def fetch_result(self):
        if self.ndim == 3:
            res = self.ocl.fetch(self.d_output)
        else:
            res = self.ocl.fetch(self.d_tmp)
        return res




def main3():
    gpu_convol3d_instance = Gpuconvol(shape=(512, 512, 512))#, device=(1, 0))
    print("Gpu 3D convolution instanciated on %s" % gpu_convol3d_instance.ocl.devicename)


    input = scipy.misc.lena().astype('f') # !
    input_tmp = np.zeros((input.shape[0], input.shape[0], input.shape[1]))
    for i in range(input.shape[0]):
        input_tmp[i, :, :] = np.copy(input) + 50*np.random.rand(input.shape[0], input.shape[1])
    input = input_tmp

    n_it = 3 # Put many iterations to check if memory is OK
    for i in range(n_it):
        t0 = time.time()
        gpu_convol3d_instance.gaussian_filter(input, 1.2)
        print("GPU Convolution %d took %.3f ms" % (i+1, (time.time()-t0)*1e3))

    res = gpu_convol3d_instance.fetch_result()
    t0 = time.time()
    ref = scipy_gaussianfilter(input, 1.2)
    print("Scipy Convolution took %.3f ms" % ((time.time()-t0)*1e3))
    print("Max diff : %f" % np.max(np.abs(res - ref)))





def main():

    # Inputs : 3D image and Gaussian kernel
    input = scipy.misc.lena()
    input_tmp = np.zeros((input.shape[0], input.shape[0], input.shape[1]))
    for i in range(input.shape[0]):
        input_tmp[i, :, :] = np.copy(input)
    input = input_tmp
    # "weird" size
    #~ input = input[0:354, 0:507,0:209]
    im_z, im_h, im_w = input.shape
    sigma = 2.0
    gaussian = comp_kern_scipy(sigma)
    ksize = gaussian.shape[0]


    # Initialize device
    ocl = Ocl(profile=True)
    print("running on %s" % ocl.devicename)

    # Build Program
    ocl.add_program_path("opencl/")
    program = ocl.compile_file("convolution.cl")

    t0 = time.time()
    # Device memory
    d_input = ocl.to_device(input, flags="r")
    d_gaussian = ocl.to_device(gaussian, flags="r")

    d_output = ocl.create_buffer_like(input, flags="w")
    d_tmp = ocl.create_buffer_like(input)


    # Execute
    wg = None # unspecified workgroup size
    shape = (im_w, im_h, im_z)
    k1 = ocl.call(program.horizontal_convolution, shape, wg, d_input, d_tmp, d_gaussian, ksize, im_w, im_h, im_z)
    k2 = ocl.call(program.vertical_convolution, shape, wg, d_tmp, d_output, d_gaussian, ksize, im_w, im_h, im_z)

    # Retrieve result
    res = ocl.fetch(d_output)
    print("Horizontal convolution took %.3fms and vertical convolution took %.3fms" % (1e-6 * (k1.profile.end - k1.profile.start),
                                                                                          1e-6 * (k2.profile.end - k2.profile.start)))
    t1 = time.time()
    print("GPU : %.3f ms" % ((t1 - t0)*1e3))


    # Scipy version
    t1 = time.time()
    ref = scipy_gaussianfilter(input, sigma)
    print("scipy.ndimage.gaussian_filter : %f ms" % ((time.time() - t1)*1e3))

    # Check
    print("Max diff : %f" % np.max(np.abs(res - ref)))



if __name__ == '__main__':

    main()

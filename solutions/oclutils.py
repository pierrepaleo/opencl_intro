#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pierre Paleo <pierre.paleo@esrf.fr>
#  License: BSD
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of ESRF nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE


import numpy as np
import os
import pyopencl as cl
import pyopencl.array as parray



def next32(i):
    """
    Round a number up to the next multiple of 32
    """
    while (i & 31):
        i += 1
    return i




class Ocl:
    """
    Simple wrapper for OpenCL, providing :
        - Device initialization (platform, device, context, queue)
        - Array check (type, contiguous memory layout, C order)
        - Grid/block size computation
        - Memory allocation/transfers
        - Guess data type of device buffers, by book-keeping host->device transfers and buffers creations
    """


    def __init__(self, profile=False, device=None, manual=False):
        """
        Initialize a device, a context and a queue.
        The preferred device is a NVIDIA GPU with maximum compute capability.

        @param profile : (optional) if True, enable profiling of the OpenCL events
        @param device : (optional) device in the format (0, 0)
        @param manual : (optional) if True, choose manually a device from the PyOpenCL prompt.
        """
        platforms = cl.get_platforms()

        if manual:
            self.ctx = cl.create_some_context()
            self.device = ctx.devices[0]

        elif device:
            self.device = platforms[device[0]].get_devices()[device[1]]
            self.ctx = cl.Context([self.device])

        else:
            # Try to choose a NVIDIA card with best compute capability
            cc_max = -1
            cc_argmax = (0, 0)
            for i_p, p in enumerate(platforms):
                for i_dev, dev in enumerate(p.get_devices()):
                    try:
                        cc = dev.compute_capability_major_nv + 0.1 * dev.compute_capability_minor_nv
                        if cc > cc_max:
                            cc_max = cc
                            cc_argmax = (i_p, i_dev)
                    except:
                        pass
            if cc_max == -1:
                print("Warning: could not find a NVIDIA card. Please pick up manually the target device")
                self.ctx = cl.create_some_context()
                self.device = ctx.devices[0]
            else:
                self.device = platforms[cc_argmax[0]].get_devices()[cc_argmax[1]]
                self.ctx = cl.Context([self.device])
            # ------------
        self.devicename = self.device.name
        if profile:
            self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        else:
            self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.path = []
        self.book = {}


    @staticmethod
    def check_array(arr):
        """
        Check an array before sending to the GPU :
            - data type (64b is converted to 32b)
            - memory layout (contiguous)
            - C order
        @param arr : numpy ndarray
        """
        if (arr.dtype == np.float32) or (arr.dtype == np.float64):
            target_type = np.float32
        elif arr.dtype == np.int64 or arr.dtype == np.int32:
            target_type = np.int32
        # Other 64 bits types
        elif arr.nbytes/arr.size == 8:
            target_type = np.float32
        else: # TODO : what to do in that case ?
            target_type = a.dtype
        return arr.astype(target_type) if arr.flags['C_CONTIGUOUS'] else np.ascontiguousarray(arr.astype(target_type))


    @staticmethod
    def calc_size(shape, blocksize):
        """
        Calculate the optimal size for a kernel according to the workgroup size
        """
        if "__len__" in dir(blocksize):
            return tuple((int(i) + int(j) - 1) & ~(int(j) - 1) for i, j in zip(shape, blocksize))
        else:
            return tuple((int(i) + int(blocksize) - 1) & ~(int(blocksize) - 1) for i in shape)



    def oclflags(self, default, flags=None):
        """
        Utility to transform human-readable flags "r", "w", "rw" to OpenCL flags.

        @param default : native flag of the target buffer (for eg. pyopencl.mem_flags.COPY_HOST_PTR)
        @param flags : additional human-readable flags : "r", "w", "rw"
        """
        clflags = default
        if flags is not None:
            flags = flags.lower()
            if flags == "r": clflags |= self.mf.READ_ONLY
            elif flags == "w": clflags |= self.mf.WRITE_ONLY
            else: clflags |= self.mf.READ_WRITE
        else: clflags |= self.mf.READ_WRITE
        return clflags


    def book_keep(self, d_id, data_size, data_type):
        """
        Function that keeps track of created/transfered buffers on the device,
        in order to guess the data type later when transfering back to host.
        The book-keeping structure is a dictionary. The keys are the buffer device id,
        and the values are a tuple (data_size, data_type) where
            - data_size can be a scalar : the size in bytes ; or a tuple : the shape
            - data_type can be None (for unknown data type) or a native Python data type
        """
        self.book[d_id] = (data_size, data_type)


    def to_device(self, arr, destbuf=None, flags=None):
        """
        Transfer a numpy array to device.
        Returns the id of the device array.

        @param arr : numpy ndarray
        @param destbuf : (optional) destination buffer on device, already allocated
        @param flags : (optional) memory flags : "r", "w", "rw".
        """

        clflags = self.oclflags(self.mf.COPY_HOST_PTR, flags)
        arr_c = Ocl.check_array(arr)
        if destbuf is None:
            d_id =  cl.Buffer(self.ctx, clflags, hostbuf=arr_c)
        else:
            if destbuf.size != arr_c.nbytes: # TODO : only throw error for destbuf.size < arr.nbytes ? But what would be the behavior ?
                raise ValueError("ERROR: to_device(): requested to transfer an array of %d bytes, when the device buffer is %d bytes" % (arr.nbytes, destbuf.size))
            try:
                ev = cl.enqueue_copy(self.queue, destbuf, arr_c)
                # check ev against  cl.command_execution_status.{COMPLETE, SUBMITTED, RUNNING, QUEUED}
                d_id = destbuf
            except cl.LogicError:
                raise RuntimeError("ERROR: to_device(): failed to transfer array of shape %s (dtype=%s)" % (str(arr.shape), str(arr.dtype)))
        self.book_keep(d_id, arr_c.shape, arr_c.dtype)
        return d_id


    def create_buffer(self, shape, dtype, flags=None):
        """
        Allocate a buffer on the device.

        @param shape : shape (in numpy ndarray sense) of the buffer to be created
        @param dtype : numeric type (for eg. np.float32)
        @param flags : memory access flags : "r", "w", "rw"
        """
        clflags = self.oclflags(self.mf.ALLOC_HOST_PTR, flags)
        nbytes = np.empty(shape, dtype=dtype).nbytes
        d_id = cl.Buffer(self.ctx, clflags, nbytes)
        self.book_keep(d_id, shape, dtype) # possibly unknown data type
        return d_id


    def create_buffer_like(self, arr, flags=None):
        arr_c = Ocl.check_array(arr)
        size = arr_c.nbytes
        d_id = self.create_buffer(arr_c.shape, arr_c.dtype, flags)
        self.book_keep(d_id, arr_c.shape, arr_c.dtype)
        return d_id


    def create_buffer_zeros(self, shape, dtype, flags=None): # FIXME : flags are not supported in parray (?)
        a_id = parray.zeros(self.queue, shape, dtype)
        d_id = a_id.data
        self.book_keep(d_id, shape, dtype)
        return d_id


    def create_buffer_zero_like(self, arr, flags=None): # FIXME : flags are not supported in parray (?)
        raise NotImplementedError("Not Implemented yet !")


    def release_buffer(self, d_id):
        """
        Release a buffer from GPU memory.

        @param d_id: id of the device buffer, preferably created with
        create_buffer(), create_buffer_like(), create_buffer_zeros(), create_buffer_zero_like() or to_device().
        """

        if d_id is None:
            print("Warning: release_buffer(): attempted to release already freed buffer")
        else:
            if d_id in self.book.keys():
                _ = self.book.pop(d_id)
            try:
                d_id.release()
                #~ d_id = None
            except cl.LogicError:
                raise RuntimeError('Error while freeing buffer %s' % d_id)




    def add_program_path(self, path):
        if os.path.isdir(path):
            self.path.append(path)
        else:
            raise ValueError("ERROR: Ocl.add_program_path(): %s no such directory" % path)


    def compile_file(self, fname):
        found = 0
        if os.path.isfile(fname): found = 1
        # File not directly found, relative file name
        elif not(os.path.isabs(fname)):
            for d in self.path:
                fname2 = os.path.join(d, fname)
                if os.path.isfile(fname2):
                    fname = fname2
                    found = 1
        if found == 0:
            raise ValueError("ERROR: Ocl.compile_file() : %s not found" % fname)
        src = open(fname).read()
        return cl.Program(self.ctx, src).build()


    def fetch(self, d_id, dest=None, return_event=False):
        """
        Fetch a buffer from the device and returns a numpy array.
        The data type is guessed from the book-keeping structure of this class.
        If the data type cannot be guessed, a "dest" numpy ndarray has to be provided for the transfer target.

        @param d_id: pyopen.Buffer object, preferably created with Ocl.to_device, Ocl.create_buffer_like or Ocl.create_buffer.
        @param dest: optional, target numpy array
        @param return_event : optional, set to True if you want the OpenCL event in the return value
        @return: dest, event_res : if return_event is True : the target numpy array, and the pyopencl event associated to the transfer,
            otherwise the target numpy array.
        """
        if dest is None: # Unspecified target numpy array : have to guess the data type and shape
            if d_id in self.book.keys(): # The buffer is known in the book-keeping structure
                size, dtype = self.book[d_id]
                if dtype is None: # Unknown data type : the buffer is a "bag of bytes". Nothing can be further done
                    raise RuntimeError("ERROR: Ocl.fetch(): the data type of buffer %s is unknown. Please use the 'dest' keyword to specify a target numpy array." % d_id)
                dest = np.zeros(size, dtype=dtype)
            else: # Unknown buffer : a proper "dest" numpy array must be specified
                    raise ValueError("ERROR: Ocl.fetch(): unknown buffer %s. Please specify a target buffer for the transfer using the 'dest' keyword" % d_id)

        event_res = cl.enqueue_copy(self.queue, dest, d_id)
        if return_event:
            return dest, event_res
        else:
            return dest


    def call(self, kernel, grid, block, *args):
        """
        Helper to call a kernel :
            - Computes the grid size/block size according to the block, if provided
            - Makes sure that the arguments are passed as 32 bits (for eg xx.shape[y])

        @param kernel: a kernel associated to a compiled program : program.kernel_name
        @param grid : the grid size (number of thread per dimensions)
        @param block : the block size. Set to None if not relevant.
        @param args: the arguments passed to the kernel
        """
        if block:
            grid = Ocl.calc_size(grid, block)
        # Modifies the arguments to clean all 64b types
        args_list = list(args)
        for i, arg in enumerate(args_list):
            if type(arg) in [np.int64, int]:
                args_list[i] = np.int32(arg)
        newargs = tuple(args_list)
        # Call the kernel
        return kernel(self.queue, grid, block, *newargs)






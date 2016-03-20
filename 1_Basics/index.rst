.. raw:: html

   <!-- Patch landslide slides background color --!>
   <style type="text/css">
   div.slide {
       background: #fff;
   }
   </style>

OpenCL basics
==============

Outline
-------

TODO

----


OpenCL device, context, queue
------------------------------

* A **device** refers to a targeted physical device (GPU, multi-core CPU, accelerator)
    * The "parallelized code" is executed on the device
    * The device is controlled by a *host* (CPU)
* The device is tied to the host by one (or several) **context**
* Each context contains a **queue** of instructions
    * Each instruction is "enqueued" in the queue of a context
    * Instructions can be synchronous or asynchronous

.. figure:: ../images/context_queue_small.png
   :align: center
   :width: 500


.. code-block:: python

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    cl.enqueue_copy(queue, d_myarray, myarray)


.. notes: 
    For this example: simple copy device to host (C convention)
    Encapsulation => OOP friendly
    in PyOpenCL : no "size" provided... book keeping of buffers (np or device)


----

Threads, grid and groups
-------------------------

* A **thread** is a sequence of instructions executed by a computing unit (physical core)
* Threads can be launched by **groups** sharing cached memory
* The total number of launched threads is specified with the **grid**

.. figure:: ../images/gridblock.png
   :align: center
   :width: 450

Before calling an instruction, one has to specify

* The grid size : how many threads will be launched in total
* The work group size : how many threads are grouped

.. notes:
    CUDA => grid can be up to 3D. OpenCL => no limitation of dimension ("NDRange") in the specification
    in CUDA, the following are specified: n_blocks and n_threads_per_block
    on the Figure, note the row-major format. In CUDA: dimension 0 = x, dimension 1 = y

----

OpenCL kernels
---------------

* A *kernel* is a fundamental function executed by each thread
* All the threads execute the same kernel
    * thread-dependent instructions are specified in the kernel
* The execution order is not deterministic
    * Some synchronization routines (see later)


Some remarks on kernels code :

* A kernel has no return value (``void`` function)
* A kernel cannot call another kernel
* Always check array bounds inside kernels !

.. notes: 
    1: kernels can handle C structs, and even classes in CUDA (maybe OCL 2.x)
    2: preprocessor macros and inline functions
    3: kernels are called with grid size/block size => no check at this stage

----

Our first OpenCL kernel
------------------------

.. code-block:: C

    __kernel void add(
        __global float* arr1, 
        __global float* arr2, 
        __global float* res, 
        int N) 
    {
        int tid = get_global_id(0); // Thread ID
        if (tid < N) {
            res[tid] = arr1[tid] + arr2[tid];
        }
     }
    
.. notes: No loop ! Faire un dessin

* The kernel codes are identified with the ``__kernel`` prefix
* The ``__global`` prefix indicates that the array is on the device central memory
* In OpenCL, the kernel codes can be provided
    * As a C string (``const char*``)
    * In a separate ``.cl`` file. Recommended for readability !

----

Our first (py)OpenCL kernel launch
-----------------------------------

Now that the kernel is written, it has to be called from the *host*.























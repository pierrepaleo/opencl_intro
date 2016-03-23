.. raw:: html

   <!-- Patch landslide slides background color --!>
   <style type="text/css">
   div.slide {
       background: #fff;
   }
   </style>


2. Accessing memory in GPU computing
======================================

----

Introduction
------------

* Most of the times, performances depend on the way the memory is accessed
* GPUs come with different cache levels

**There is no memory fence mechanism !**

* Reading outside of array bounds often results in nonsense results
* Writing outside of array bounds can corrupt other memory buffers

----

Cache levels
-------------

+--------------+---------------------+------------------------------+
| Cache level  | Used by             | Typical use                  |
+==============+=====================+==============================+
| Registers    | Registers           | Threads variables            |
+--------------+---------------------+------------------------------+
| L1           | Shared memory       | Workgroup communication      |
+--------------+---------------------+------------------------------+
| L2           | Constant memory     | Look-Up Tables               |
+--------------+---------------------+------------------------------+
| X            | Global memory       | Device arrays                |
+--------------+---------------------+------------------------------+

GPU cache are not intended for the same use as CPU cache

* Cache size/thread is smaller than for CPU
* Typical use: 
    * smoothing out access patterns
    * avoiding memory spill (register -> global mem)


.. notes: see https://www.olcf.ornl.gov/wp-content/uploads/2013/02/GPU_Opt_Fund-CW1.pdf


----

Access patterns: warm-up
-------------------------

Consider the following C code. Which loop is the most efficient ?

.. code-block:: C

    // Version 1
    for (int i = 0; i < Nrows; i++) {
        for (int j = 0; j < Ncols; j++) {
            func(arr[i * Ncols + j]); // read and/or write
        }
    }
    // Version 2
    for (int j = 0; j < Ncols; j++) {
        for (int i = 0; i < Nrows; i++) { 
            func(arr[j * Nrows + i]); // read and/or write
        }
    }
    

----

Cache lines
-------------

* On CPU and GPU, memory access are cached
* C-like languages (C, C++, OpenCL, CUDA) are row-major. The first version is more efficient.

  
.. figure:: ../images/memaccess1.png
   :align: center
   :width: 400
   

* Example: On NVidia GPUs, memory is accessed by lines of 128 Bytes (32 elements of 4B)
    * Each load/store actually calls 32 memory transactions
    * This has to be taken into account when accessing memory !
    
----

Coalesced memory access
------------------------

A memory access is **coalesced** if adjacent threads access to contiguous memory locations.


.. figure:: ../images/memaccess2.png
   :align: center
   :width: 400

* This is the optimal memory access pattern for both global and shared memory
* This is not always possible
* Recent architectures have complex caching mechanisms for global memory
   
   
.. notes: constant memory => cache is automatically done
   

























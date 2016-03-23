.. raw:: html

   <!-- Patch landslide slides background color --!>
   <style type="text/css">
   div.slide {
       background: #fff;
   }
   </style>


3. Parallel patterns
=====================

----


Introduction
-------------

In our case, common memory access patterns are

* Element-wise
* Scatter
* Gather
* Reduction


.. figure:: ../images/accesspatterns.png
   :align: center
   :width: 400
   

* Other patterns: compact, map, scan, partition

.. notes: compact/expand ; map/reduce ; scan (eg. cumsum)
.. notes: https://stanford-cs193g-sp2010.googlecode.com/svn/trunk/lectures/lecture_6/parallel_patterns_1.pdf
.. notes: http://www.cs.nyu.edu/courses/fall10/G22.2945-001/slides/lect10.pdf
.. notes: https://people.cs.uct.ac.za/~jgain/lectures/Algorithms.pdf


----

Element-wise access
--------------------

* Each thread access to one memory location
* There is an "obvious" one-to-one thread-memory map (preferably coalesced)
* "Embarrassingly parallel"


Exercices
----------

1) Write a kernel performing the sum of two 2D ``float`` arrays.

2) Write a kernel performing the flat-field correction of an image.


Example: flat-field correction
-------------------------------

.. code-block:: C

    __kernel void flatfield(
        __global float* image, 
        __global float* dark,
        __global float* flat, 
        int Nr, 
        int Nc)
    {
    	float data;
    	int i = get_global_id(0);
        int j = get_global_id(1);
        int idx = i*Nc + j;
    	if(i < Nr && j < Nc) {
    	   data -= dark[i];
    	   data /= flat[i];
    	}
    }



Scatter/Gather 
---------------

* Gather: read multiple data items to a single location 
* Scatter: write a single data item to multiple locations 

Gather with special access pattern (eg. convolution) is sometimes called *stencil*

Read/write issues

* If different threads read from the same memory location, the access are *serialized*, slowing down the process.
* If different threads write to the same memory location, *behavior is unpredictable*. 

Atomic operations
------------------

* Making different threads write to the same memory location results in a conflict.
* Solution: **atomic operations**, i.e operations that cannot be interrupted

.. code-block:: C
    
    int atomic_add (int *p, int val)

**Note**: there is no atomic operations on ``float`` for OpenCL 1.x implementations !





----

Exercices
----------

1) Write a kernel performing a 2x2 binning of an image (its dimensions are assumed to be even).

2) Write a kernel performing the histogram of an image

















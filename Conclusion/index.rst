.. raw:: html

   <!-- Patch landslide slides background color --!>
   <style type="text/css">
   div.slide {
       background: #fff;
   }
   </style>

Conclusion
============

Take-home messages

* Non-negligible algorithmic work to write efficient parallel code
* On discrete GPUs, avoid Device<->Host transfers when possible
* Profile your code !



The following points were not addressed:

* Work-group size optimization
* Texture memory (another cached memory)
* Other interesting patterns like scan and map-reduce


Where to go from here:

* `pyopencl examples <https://github.com/pyopencl/pyopencl/tree/master/examples>`_
* `pyopencl tutorials <https://documen.tician.de/pyopencl/#tutorials>`_


Core Module
===========

Objective
---------
An objective function to optimize. 

Cost Function
-------------
A term in the objective function as a function of one or more :class:`Variable` objects.

Variable
--------
A variable in the optimization problem. :class:`Variable` objects are named wrappers for
``torch`` tensors.

Cost Weight
-----------
A weight for cost functions. 

Reference
---------
.. autosummary::
    :toctree: generated
    :nosignatures:

    theseus.Objective
    theseus.Variable
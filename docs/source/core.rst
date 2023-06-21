Core Module
===========

Objective
---------
An objective function to optimize (see :class:`theseus.Objective`). 

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
        :no-undoc-members:
        :nosignatures:
    theseus.Objective.add
    theseus.Objective.error
    theseus.Objective.error_metric    
    theseus.Objective.update
    theseus.Objective.retract_vars_sequence
    theseus.CostFunction
        :no-undoc-members:
        :nosignatures:
    theseus.Variable
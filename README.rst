dinpy
=====

dinpy can be used to investigate Boolean and discrete interaction networks.

A Boolean network can be created for instance as follows:

.. code:: python

      >>> from dinpy.input_din import read_truth_table
      >>> f = read_truth_table(["00 10", "10 11", "11 01", "01 10"])

A state is represented as a tuple, and a network is stored as a dictionary mapping states to states.
We can explore properties of the asynchronous dynamics or the interaction graphs:

.. code:: python

      >>> from dinpy.din import attractors
      >>> for a in attractors(f):
      ...     print(a)
      ...
      set([(0, 1), (1, 0), (0, 0), (1, 1)])
      >>> from dinpy.interaction_graphs import local_int_graphs, global_circuits
      >>> local_int_graph(f)
      {(0, 1): [(1, 1, -1), (1, 2, 1)], (1, 0): [(1, 2, 1), (2, 1, -1)], (0, 0): [(1, 2, 1)], (1, 1): [(1, 1, -1), (1, 2, 1), (2, 1, -1)]}
      >>> global_circuits(f)
      [([1], -1), ([1, 2], -1)]

Multilevel networks can be converted to Boolean networks using the methods of `Tonello (2017-) <https://arxiv.org/abs/1703.06746>`_ or `Fauré and Kaji (2018) <https://www.sciencedirect.com/science/article/pii/S0022519317305532>`_.

.. code:: python

      >>> from dinpy.multi_to_boolean import multi_to_boolean, binarisation
      >>> from dinpy.din import to_stepwise, to_asymptotic
      >>> f = {(0,): (2,), (1,): (1,), (2,): (1,)}
      >>> multi_to_boolean(f)
      {(0, 1): (1, 0), (1, 0): (1, 0), (0, 0): (1, 1), (1, 1): (1, 0)}
      >>> multi_to_boolean(to_stepwise(f))
      {(0, 1): (1, 0), (1, 0): (1, 0), (0, 0): (1, 0), (1, 1): (1, 0)}
      >>> binarisation(to_asymptotic(f))
      {(0, 1): (0, 1), (1, 0): (1, 0), (0, 0): (1, 1), (1, 1): (0, 0)}

We can also generate Boolean networks with some desired characteristics,
for instance, a given interaction circuit and some fixed states:

.. code:: python

      >>> from dinpy.find_din import boolean_map, is_global_circuit, fixed_point, solve
      >>> from pysmt.shortcuts import And
      >>> f = boolean_map(2)
      >>> next(solve(And(is_global_circuit(f, [1,2], sign=+1), fixed_point(f, (0, 1))), 2))
      {(0, 1): (0, 1), (1, 0): (1, 0), (0, 0): (0, 1), (1, 1): (0, 0)}

For other examples, see `examples </examples>`_ or `tests </tests>`_.

The scripts are intended for exploration of small networks, with up to around 10 variables.
For networks with more variables, we recommend considering `GINsim <http://ginsim.org/>`_ or `PyBoolNet <https://github.com/hklarner/PyBoolNet/>`_.

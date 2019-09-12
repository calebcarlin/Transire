.. module:: source.surface_splitter

Surface Split Search
======================

This method allows for predicting the cleavage structure
of interfaces by separating the two sides of the
interface with a vacuum and then varying the positions
of the interfacial atoms.  The plane of the interface
is defined during the interface construction process
or by the user when reading in a structure.  The atoms
included in the search are designated by the user by 
specifying the an interfacial region relative to the
plane of the interface.  The movement of the atoms
are constrained to moving parallel to the z-axis,
displaced the same distance as was used to separate
the two sides of the interface structure.

The discrete nature of the above arrangement means
that the state of the separated structure and any
subsequent structures as bit-arrays where each
interfacial atom is assigned a bit and the 1 and 0
corresponding to the atom being on the upper side and
lower side of the separation respectively.  For ease of
record keeping, the bit-string is converted to an integer
for the purpose of file-names and other outputs. There 
are three options supported at present for sampling
the cleavage configurations; random sampling,
Gaussian Process Regression (GPR), and a single-pass
relaxation.  

The random sampling method involves three steps:
perturb the state, check if state has already been
sampled, and calculate the energy of the new state
and add to the collected data set of states and 
corresponding energies.  The perturbation is either
flipping a randomly chosen interfacial atom or by
selecting a random state from all of the possible
states.  Aside from ensuring that the random walk is
self-avoiding, there is no criteria used to accept or
reject each of the investigated states.

The GPR uses the data compiled of states and
corresponding energies and a kernel based on the
radial basis function (RBF) kernel where the distance
between two states is defined by Sokal-Michener metric.
The expected improvement is used as a surrogate model
to determine the next state to be added into the data
set and to subsequent calculations of the GPR function.
Currrently, the process of determining new states and
then incorporating the energy of those states continues
until either the number of user specified steps
is exhausted or the std_deviation drops to 0 as the
number of data points used to fit the GPR gets too
large.

The discrete nature of the surface flipping routine
results often in the GPR locating states near the
minima, but separated by a small number of atoms
that are not in the optimal positions.  To address
this, the single-pass relaxation can be performed
on a user specified number of lowest energy
states located by the random walk and/or GPR methods.
The single-pass method flips each interfacial atom
and calculates the energy.  If the energy of the new
state is the lowest energy of the current pass, then
the change is accepted.  Otherwise, the flip is
reverted.  The process continues until all of the
interfacial atoms have been flipped once, and then
the lowest energy and corresponding state for that
pass 



____________

.. autoclass:: surface_atom_flip

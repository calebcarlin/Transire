# copyright Caleb Michael Carlin (2018)
# Released uder Lesser Gnu Public License (LGPL)
# See LICENSE file for details.

import ase
from ase import Atoms, Atom
import numpy as np
from numpy.linalg import norm
import itertools
import fractions
from math import pi, floor
from ase.build import cut, make_supercell
from ase.build import stack, surface
from .utilities import lcm, printx, angle_between, surface_from_ase
from .utilities import almost_zero
from .utilities import InterfaceConfigStorage as ICS
from ase.io import write as ase_write
from ase.io import read as ase_read
from traceback import print_exc
import sys
from .spiral import Hyperspiral
import warnings


class InterfaceSupercell(object):
    """
    Class for generating interface structures from two unit cells.
   
    Parameters:

    unit_cell_a: ASE atoms object
        unit cell atoms object for the bottom side of the interface
    unit_cell_b: ASE atoms object
        unit cell atoms object for the top side of the interface
    input: InputReader object
        object with read in keywords
    """

    def __init__(self, unit_cell_a, unit_cell_b, input):
        self.raw_cell_a = unit_cell_a
        self.raw_cell_b = unit_cell_b
        self.cut_cell_a = unit_cell_a
        self.cut_cell_b = unit_cell_b
        self.layers_a = int(input.dict['crys_a_layers'])
        self.layers_b = int(input.dict['crys_b_layers'])
        self.surface_a = input.dict['crys_a_surface']
        self.surface_b = input.dict['crys_b_surface']
        self.interface = None
        self.distance = float(input.dict['separation'])
        self.super_cell_a = None
        self.super_cell_b = None
        self.input = input
        self.duplicates = []
        self.ase_version = self.determine_version()

    def cut_surface(self, surface_a, surface_b):
        """
        cut the raw unit cell to produce the cut unit cell
        """
        if self.input.dict['read_in_structure'] != 'False':
            return
        else:
            self.cut_cell_a = surface_from_ase(
                self.raw_cell_a, surface_a, layers=1)
            self.cut_cell_b = surface_from_ase(
                self.raw_cell_b, surface_b, layers=1)

    def generate_interface(self):
        """
        main function to generate interface from two cut_cells.
        Returns .True. if an error occured, otherwise returns .False.
        """

        #check to see if we need to build structure or just read in
        if self.input.dict['read_in_structure'] != 'True':
            # set up the output file so we can put the error messages
            # in the output file
            file = self.input.dict['output_file']
    
            # copy over the atom units so that cut_cells are unchanged
            unit_cell_a = self.cut_cell_a.copy()
            unit_cell_b = self.cut_cell_b.copy()
    
            # =====Debug=====
            if (self.input.dict['print_debug'] != 'False'):
                printx("========Starting Cell 1========")
                printx(str(unit_cell_a.cell))
                printx("atoms = " + str(len(unit_cell_a)))
                printx("========Starting Cell 2========")
                printx(str(unit_cell_b.cell))
                printx("atoms = " + str(len(unit_cell_b)))
    
            # replicate the unit cells so that periodicity is always preserved
            periodic_cell_a, max_coeff_a = self.protect_periodicity(unit_cell_a)
            periodic_cell_b, max_coeff_b = self.protect_periodicity(unit_cell_b)
   
            # populate the new cell using cookie cutter method on generated lattice
            try:
                self.super_cell_a = self.populate_new_cell(
                    unit_cell_a, periodic_cell_a, max_coeff_a)
                self.super_cell_b = self.populate_new_cell(
                    unit_cell_b, periodic_cell_b, max_coeff_b)
            except Exception as err:
                [printx(x) for x in err.args]
                raise Exception("Too many atoms, skipping to next step")
            
            #apply translations if the user specified them.
            self.super_cell_a.translate(self.input.dict['translate_crys_a']+[0])
            self.super_cell_a.wrap()
            self.super_cell_b.translate(self.input.dict['translate_crys_b']+[0])
            self.super_cell_b.wrap()
    
            # =====Debug=====
            if (self.input.dict['print_debug'] != 'False'):
                printx("========Ortho Cell 1========")
                printx(str(self.super_cell_a.cell))
                printx("atoms = " + str(len(self.super_cell_a)))
                printx("========Ortho Cell 2========")
                printx(str(self.super_cell_b.cell))
                printx("atoms = " + str(len(self.super_cell_b)))
    
            # calculate the smallest supercells needed to minimize
            # stress in the interface
            P_list, R_list = self.generate_interface_transform(
                self.super_cell_a, self.super_cell_b)
            P_tuple = tuple(P_list + [int(self.input.dict['crys_a_layers'])])
            R_tuple = tuple(R_list + [int(self.input.dict['crys_b_layers'])])
            # generate new supercells
            try:
                self.super_cell_a *= P_tuple
            except Exception as err:
                raise Exception(
                    "Error in generating supercell_a in interface step")
            try:
                self.super_cell_b *= R_tuple
            except Exception as err:
                raise Exception(
                    "Error in generating supercell_b in interface step")
    
            # =====Debug=====
            if (self.input.dict['print_debug'] != 'False'):
                printx("Replication A = " + str(P_tuple))
                printx("Replication B = " + str(R_tuple))
    
            # check that total size isn't too big before we continue
            total = len(self.super_cell_a) + len(self.super_cell_b)
            if (total >= int(self.input.dict['max_atoms'])):
                raise Exception("Error: interface is too large: " + str(total))
    
            # tag the two supercells so that they can be separated later
            self.super_cell_a.set_tags(1)
            self.super_cell_b.set_tags(2)
    
    
            # add a vacuum between the layers.
            if (self.distance is not None):
                self.super_cell_a.cell[2, 2] += self.distance
    
            # =====Debug=====
            if (self.input.dict['print_debug'] != 'False'):
                printx("========Super Cell 1========")
                printx(str(self.super_cell_a.cell))
                printx("atoms = " + str(len(self.super_cell_a)))
                printx("========Super Cell 2========")
                printx(str(self.super_cell_b.cell))
                printx("atoms = " + str(len(self.super_cell_b)))
    
            # stack the supercells on top of each other and set pbc to xy-slab
            try:
                self.interface, self.super_cell_a, self.super_cell_b = stack(
                    self.super_cell_a, self.super_cell_b,
                    output_strained=True, maxstrain=None)
            except Exception as err:
                raise Exception(
                    "Error in generating interface during the stack step")
    
            # set pbc to infinite slab or fully periodic setting
            if (self.input.dict['full_periodicity'] != 'False'):
                self.interface.pbc = [1, 1, 1]
            else:
                self.interface.pbc = [1, 1, 0]
    
            #add explicit vacuum above and below 
            if (self.input.dict['z_axis_vacuum'] != '0.0'):
                self.interface = self.z_insert_vacuum(self.interface)
    
        else:
        #if we are reading in a structure, then we get to skip all
        #the previous stuff
            self.read_in_structure()

        return

    def match_sizes(self, cell_side_a, cell_side_b):
        """
        the unit cells must be replicated an integer number of times.
        Using a back and forth method, the two integers are determined that
        reduces the difference between the two values is less
        than the tolerance given in the input.
        """
        a = 1.0
        b = 1.0
        convergence = cell_side_a / cell_side_b
        upper_bound = 1.0 + float(self.input.dict['tolerance'])
        lower_bound = 1.0 - float(self.input.dict['tolerance'])
        while ((convergence < lower_bound) or (convergence > upper_bound)):
            if (cell_side_a * a) < (cell_side_b * b):
                a += 1.0
                convergence = (cell_side_a * a) / (cell_side_b * b)
            else:
                b += 1.0
                convergence = (cell_side_a * a) / (cell_side_b * b)

        return a, b

    def generate_interface_transform(self, unit_cell_a, unit_cell_b):
        """
        A pair of lists for replicating the two cells so that they match
        up are generated.
        """
        P_list = []
        R_list = []

        for j in range(2):
            side_a = unit_cell_a.cell[j][j]
            side_b = unit_cell_b.cell[j][j]
            x, y = self.match_sizes(side_a, side_b)

            P_list.append(abs(int(x)))
            R_list.append(abs(int(y)))

        return P_list, R_list

    def protect_periodicity(self, unit_cell):
        """
        determines the number of copies of unit cell along each axis are
        needed to ensure that any further replication of the supercell will
        accurately match up the cells.
        """
        new_cell = unit_cell.cell.copy()

        max_coeff = [0, 0, 0]
        for i, j in [[0, 1], [0, 2], [1, 2]]:
            new_cell, max_coeff = self.twod_matching(i, j, new_cell, max_coeff)

        # if the cell has been rotated so that the vector lies on the negative
        # axis, then we need to for it to have a non-zero max_coeff
        #max_coeff = [x+1 for x in max_coeff if x == 0 else x]
        max_coeff = [x+1 if x == 0 else x for x in max_coeff]

        return new_cell, max_coeff

    def twod_matching(self, axis1, axis2, matrix, max_coeff):
        """
        Take a two by two matrix that is the projection of two lattice
        vectors on a plane and determine the retangular representation of
        the matrix.
        """

        a_b = np.array([matrix[axis1, axis1], matrix[axis1, axis2]])
        b_a = np.array([matrix[axis2, axis1], matrix[axis2, axis2]])
        coeff_storage = [[0, 0], [0, 0]]

        intersect = [0, 0]
        # if the two numbers we are looking at are not zero,
        # we find the integer multiples needed to get a zero intercept
        if not(almost_zero(a_b[1]) and almost_zero(b_a[0])):
            for i in range(2):
                if not almost_zero(a_b[i] * b_a[i]):
                    c, d = self.match_sizes(abs(a_b[i]), abs(b_a[i]))
                    if np.sign(a_b[i]) == np.sign(b_a[i]):
                        c *= -1
                else:
                    if almost_zero(a_b[i]):
                        c, d = 1, 0
                    else:
                        c, d = 0, 1
                coeff_storage[i] = [abs(int(c)), abs(int(d))]
                b = (i + 1) % 2
                intersect[b] = c * a_b[b] + d * b_a[b]
            # store the values of the zero intercept
            matrix[axis1, axis1] = abs(intersect[0])
            matrix[axis2, axis2] = abs(intersect[1])
            matrix[axis1, axis2], matrix[axis2, axis1] = 0.0, 0.0
            # store the max coefficient for later when we populate the
            # the new orthonormal cell
            max_1_coeff = coeff_storage[0][0] + coeff_storage[1][0]
            max_2_coeff = coeff_storage[0][1] + coeff_storage[1][1]
            max_coeff[axis1] = max(abs(max_1_coeff), max_coeff[axis1])
            max_coeff[axis2] = max(abs(max_2_coeff), max_coeff[axis2])

        return matrix, max_coeff

    def check_zero_diag(self, cell_matrix):
        """
        check if any diagonal elements are zero.  If there are any, swap
        the rows around until the check is passed.
        """
        while True:
            for i in range(3):
                if almost_zero(cell_matrix[i, i]):
                    next_val = (i + 1) % 3
                    cell_matrix = self.swap_rows(cell_matrix, i, next_val)
                    break
            # only way we get here is if all the diagonals are non-zero
            return cell_matrix

    def swap_rows(self, cell, row1, row2):
        """
        swap two rows in an array.
        """
        cell[row1, :], cell[row2, :] = cell[row2, :].copy(), cell[row1,
                                                                  :].copy()

        return cell

    def translate_cell(self, cut_cell, translation):
        """
        translate the atoms in the cell and then wrap back into the cell.
        """
        cut_cell.translate(translation)
        cut_cell.wrap(pbc=(1, 1, 0))

        return cut_cell

    def rotate_cell(self, cut_cell, rotation):
        """
        rotate the atoms and the cell vectors.
        """
        cut_cell.rotate(a=rotation, v=self.input.dict['angle_axis'],
                        rotate_cell=True)

        return cut_cell

    def determine_version(self):
        """
        determine the version of ase being used since the update after 3.12
        the way of building structures changed.
        """

        asever = (ase.__version__).split('.')

        return int(asever[1])

    def in_new_cell(self, atom, cell, error):
        """
        quick function to see an atom is inside a cell with the given error.
        """
        if (atom[0] < -error) or (atom[0] > (cell[0, 0] - error)):
            return False
        if (atom[1] < -error) or (atom[1] > (cell[1, 1] - error)):
            return False
        if (atom[2] < -error) or (atom[2] > (cell[2, 2] - error)):
            return False
        return True

    def populate_new_cell(self, unit_cell, new_cell, max_coeff):
        """
        Fill up an orthorhombic cell wiht the atoms from a unit cell.
        Each atom is translated by a multiple of the old lattice vectors,
        and accepted atoms are added to the new object until the atom density
        matches that of the unit cell.
        """

        super_cell = Atoms()
        super_cell.set_cell(new_cell)
        # setup storage for rejected atoms in case we need them
        rejects = Atoms()
        volume = unit_cell.get_volume()
        new_volume = super_cell.get_volume()
        atoms = int(round(float(len(unit_cell)) * new_volume / volume))
        # quick check to see if the new cell will have too many atoms
        if (atoms > int(self.input.dict['max_atoms'])):
            raise Exception("too many atoms in supercell")
        vectors = np.asarray(unit_cell.cell)
        spiral = Hyperspiral(max_coeff)
        atom_positions = unit_cell.get_positions()
        # have to zero out infinitesimal values in atom_positions

        # =====Debug=====
        if self.input.dict['print_debug'] != "False":
            printx("old cell = " + str(unit_cell.cell))
            printx("new cell = " + str(new_cell))
            printx("max_coeff = " + str(max_coeff))
            printx("atoms = " + str(atoms))

        # move through the representations of the initial unit cell along a
        # spiral pattern on a grid of integer values.  first the spiral on
        # a plane is completed and then the spiral is shifted down and then up
        # along the third coordinate.
        while True:
            shift = np.matmul(spiral.position, vectors)
            for i in range(len(unit_cell)):
                atom_prime = np.add(shift, atom_positions[i])
                #check if the new atom is in the cell and make sure
                #that there isn't already an atom at the new position
                if (self.in_new_cell(atom_prime, new_cell, 1.0e-9)
                     and not np.any(
                     np.equal(atom_prime,super_cell.positions).all(axis=1))):
                    new_atom = unit_cell[i]
                    new_atom.position = atom_prime
                    super_cell.append(new_atom)
                    atoms -= 1
                    # satisfying this condition means success
                    if atoms == 0:
                        return super_cell
                else:
                    new_atom = unit_cell[i]
                    new_atom.position = atom_prime
                    rejects.append(new_atom)

            # if we get to the end of the spirals then we check
            # the edges for barely rejected atoms to add in
            try:
                spiral.tick()
            except Exception as err:
                [printx(x) for x in err.args]
                if self.input.dict['print_debug'] != 'False':
                    print_exc()
                try:
                    super_cell = self.check_edges(
                        rejects, new_cell, super_cell, atoms)
                except Exception as err:
                    raise Exception(err.args[0])
                return super_cell

        return super_cell

    def check_edges(self, rejects, new_cell, super_cell, atoms):
        """
        go through the rejected atoms to find one that is close enough to our
        boundries that we can add it in for edge cases.
        """
        for i in range(len(rejects)):
            if self.in_new_cell(rejects[i].position, new_cell, 1e-3):
                super_cell.append(rejects[i])
                atoms -= 1
                if atoms == 0:
                    return super_cell

        # if we get here, then we have failed to make the super_cell
        raise Exception("Error: failed to populate the cell")

        return super_cell

    def z_insert_vacuum(self, interface):
        """
        Add a vacuum above and below the crystal slabs by editing cell
        and shifting atoms.
        """

        vacuum = float(self.input.dict['z_axis_vacuum'])/2.0

        interface.cell[2][2] += vacuum*2

        for x in range(len(interface)):
            interface.positions[x,2] += vacuum

        return interface

    def read_in_structure(self):
        """
        Read in a coordinate file and construct a completed interface
        structure by splitting it in two and populating the ICS.
        We have to use fake unit cells as they may not be applicable.
        """

        structure_file = self.input.dict['read_in_file']
        try:
            new_atom = ase_read(structure_file)
        except Exception as err:
            printx("Error when reading in interface structure")
            raise Exception(err.args[0])

        #get vertical position of lowest and highest atoms in the cell
        low_val = np.amin(new_atom.get_positions()[:,2])-0.02
        high_val = np.amax(new_atom.get_positions()[:,2])+0.02
        interface_slice = float(self.input.dict['read_in_sep_plane'])
        #we already checked to make sure that either both are None 
        #or neither is
        if self.input.dict['read_in_crys_a_layer_depth'] is not None:
            a_depth = float(self.input.dict['read_in_crys_a_layer_depth'])
            b_depth = float(self.input.dict['read_in_crys_b_layer_depth'])
        #rough guess so we can code below.  Highly recommend users
        #give the values
        else:
            a_depth = new_atom.cell[2,2]/8.0
            b_depth = a_depth

        #remove vacuum on the extremities by shifting down the atoms
        #and setting the upper bound of the cell
        #shift the other slcing planes to match
        if self.input.dict['read_in_exclude_vacuum'] != 'False':
            new_atom.positions[:,2] -= low_val
            new_atom.cell[2,2] = high_val - low_val - 0.02
            interface_slice -= low_val + 0.01

        #set up the mock-unit cell objects.  We want the 
        #XY dimensions from the full interface, while setting
        #the Z dimension to the predefined layer depth
        new_a_unit = Atoms(pbc = True, cell=new_atom.cell)
        new_a_unit.cell[2,2] = a_depth
        new_b_unit = Atoms(pbc = True, cell=new_atom.cell)
        new_b_unit.cell[2,2] = b_depth

        for i in new_atom:
            if i.position[2] < (a_depth + low_val):
                new_a_unit.append(i)
            if i.position[2] > (high_val - b_depth):
                new_b_unit.append(i)
            if i.position[2] < interface_slice:
                i.tag = 1
            if i.position[2] >= interface_slice:
                i.position[2] += self.distance
                i.tag = 2

        new_atom.cell[2,2] += self.distance

        #populate the class objects
        self.interface = new_atom.copy()
        # set pbc to infinite slab or fully periodic setting
        if (self.input.dict['full_periodicity'] != 'False'):
            self.interface.pbc = [1, 1, 1]
        else:
            self.interface.pbc = [1, 1, 0]
        self.cut_cell_a = new_a_unit.copy()
        self.cut_cell_b = new_b_unit.copy()

        return

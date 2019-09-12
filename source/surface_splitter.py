# copyright Caleb Michael Carlin (2019)
# Released under Lesser GNU Public License (LGPL)
# See LICENSE file for details

from ase import Atoms
from .energy_calculators import EnergyCalc, calculate_binding_energy
from .utilities import InterfaceConfigStorage as ICS
from .utilities import write_xyz, printx
import numpy as np
import scipy as sp
import random
import math
import os
import sys
from scipy.spatial.distance import pdist,cdist,squareform
from scipy.stats import norm
from scipy.optimize import minimize

class surface_atom_flip(object):
    """
    method for determining work of separation for an interface 
    by separating the slabs and then flipping the atoms at the
    interface back and forth to find the lowest energy break
    as well as the ensemble of surface states.
    
    Parameters:

    input: InputReader object
        object with read in keywords
    
    ics: InterfaceConfigStorage object
        object containing the interface to be manipulated
    """

    def __init__(self, input, ics):
        self.input = input
        self.structure = ics
        self.separation = float(input.dict['flip_separation'])
        self.state_memory = []
        self.output_file = self.input.dict['output_file']
        self.max_bit = 0
        self.current_state = 0
        ics = self.flip_interface()


    def flip_interface(self):
        #determine the atoms that are in the interface region
        #set the initial state in the memory
        split_int = self.structure.copy()
        int_array = self.create_interface_array(split_int.atom)
        random.seed()
        #shift the atoms with tag 2 (upper crystal) along the z-axiz
        for i in split_int.atom:
            if i.tag == 2:
                i.position[2] += float(self.input.dict['flip_separation'])
        split_int.atom.cell[2][2] += float(self.input.dict['flip_separation'])
        
        #check if this is a restart run to see if we need to read in previous
        #memory of visited states
        if self.input.dict['flip_restart'] != 'True':
            #need to recalculate the energy of the separated, but unflippted
            #interface structure, and set as first entry in memory
            try:
                split_int.file_name = (self.input.dict['flip_file_name']+
                                    "."+str(self.b_to_i(int_array)))
                EnergyCalc(self.input,self.input.dict['energy_method'],
                        split_int,'run')
            except Exception as err:
                [printx(x) for x in err.args]
            self.state_memory = np.array([[self.b_to_i(int_array),
                                       split_int.energy]])
            write_xyz(split_int)
        else:
            self.state_memory = np.loadtxt(self.input.dict['flip_file_name']+
                                '.state_memory.out')
            int_array, split_int = self.restore_state(int_array,
                              int(self.state_memory[-1,0]),split_int)

        #check to make sure that we aren't asking for more states
        #than are possible and that we haven't reached the total number
        #of states
        random_states = int(self.input.dict['flip_n_random_states'])
        total_states = int(self.input.dict['flip_n_total_states'])
        if total_states >= (2**len(int_array))+len(self.state_memory)-1:
            total_states = 2**len(int_array)+len(self.state_memory)-2
            if random_states > total_states: random_states = total_states
        self.current_state = len(self.state_memory) - 1
        if self.current_state >= total_states:
            printx('Warning, total number of flipping states already met')

        #loop over the random perturbation of the state and calculating
        #the new energies to build up a database of states
        while self.current_state < random_states:
            printx("current state step is %i" % self.current_state)
            int_array, split_int, success = (
                   self.perturb_state(int_array,split_int,
                   self.input.dict['flip_rand_method']))
            if success:
                self.current_state += 1
                try:
                    EnergyCalc(
                        self.input, self.input.dict['energy_method'],
                        split_int, 'run')
                except Exception as err:
                    [printx(x) for x in err.args]
                self.state_memory = np.append(self.state_memory,
                    [[self.b_to_i(int_array),split_int.energy]],0)
                split_int.file_name = (self.input.dict['flip_file_name']+
                                    "."+str(self.b_to_i(int_array)))
                write_xyz(split_int)
                np.savetxt(
                        self.input.dict['flip_file_name']+'.state_memory.out',
                        self.state_memory)

        #loop over the GPR steps in a directed search
        while self.current_state < total_states:
            printx("current state step is %i" % self.current_state)
            self.current_state += 1
            gpr = self.gpr_state(int_array,split_int.atom)
            #new_x is a bit array
            new_x = self.next_state(self.expected_improvement,gpr)
            new_int = self.b_to_i(new_x)
            if new_int in self.state_memory:
                print('already tested '+str(new_int))
                pass
            else:
                int_array,split_int = self.restore_state(
                                        int_array,new_int,split_int)
                try:
                    EnergyCalc(
                        self.input, self.input.dict['energy_method'],
                        split_int, 'run')
                except Exception as err:
                    [printx(x) for x in err.args]
                self.state_memory = np.append(self.state_memory,
                    [[new_int,split_int.energy]],0)
                split_int.file_name = (self.input.dict['flip_file_name']+
                                    "."+str(new_int))
                write_xyz(split_int)
                np.savetxt(
                        self.input.dict['flip_file_name']+'.state_memory.out',
                        self.state_memory)

        lowest_energy = np.argmin(self.state_memory[:,1])
        printx("\n=====================",self.output_file)
        printx("lowest energy = "+str(self.state_memory[lowest_energy,1]),
               self.output_file)
        printx("state = "+str(int(self.state_memory[lowest_energy,0])),
               self.output_file)

        #print out final details.  If we had a sigma=0 error, then
        #we need to initialize gpr first so we can get the std-dev
        #of the lowest energy state
        #We skip this if there are no GPR steps
        if not (total_states <= random_states):
            if (self.current_state >= total_states and 
                self.input.dict['flip_restart'] != 'False'):
                gpr = self.gpr_state(int_array,split_int.atom)
            if total_states > random_states:
                X = self.i_to_b(int(self.state_memory[lowest_energy,0]))
                mu, sigma = gpr.predict(np.atleast_2d(X),return_std=True)
                printx("predicted lowest energy = "+str(mu),self.output_file)
                printx("std dev = "+str(sigma),self.output_file)

        if int(self.input.dict['flip_n_final_reduction']) >= 1:
            self.final_reductions(int_array,split_int)
            
        self.print_results()

        return split_int

    def create_interface_array(self, interface):

        #Here we find the lowest z-value for the upper slab and
        #the highest z-value for the lower slab.  The upper and
        #lower here refers to the slab.
        upper_limit = interface.cell[2,2]
        lower_limit = 0
        interface_array = []
        lower_depth = float(self.input.dict['flip_a_depth'])
        upper_depth = float(self.input.dict['flip_b_depth'])
        count = [0,0]
        
        for i in interface:
            if i.tag == 1:
                lower_limit = np.amax([i.position[2],lower_limit])
            else:
                upper_limit = np.amin([i.position[2],upper_limit])

        A = [lower_limit,lower_limit-lower_depth,upper_limit,
             upper_limit+upper_depth]
        for j in range(len(interface)):
            if (interface[j].position[2] > A[1] and 
               interface[j].position[2] < A[0]):
                interface_array.append([j, interface[j].tag-1])
                count[0] += 1
            if (interface[j].position[2] > A[2] and 
               interface[j].position[2] < A[3]):
                interface_array.append([j, interface[j].tag-1])
                count[1] += 1

        printx("Atoms in lower interfacial region: "+str(count[0]),
               self.output_file)
        printx("Atoms in upper interfacial region: "+str(count[1]),
               self.output_file)
        printx("Number of possible states: "+str(2**(count[0]+count[1])),
               self.output_file)
        #we catch this here to pad the leading 0's in i_to_b
        self.max_bit = count[0]+count[1]

        return np.array(interface_array)

    def b_to_i(self,array):
        #take the list of state integers and convert it into a
        #int.  Sadly, we must convert the array into a string,
        #then ints.
        if array.ndim == 2:
            return int("".join(str(int(x)) for x in array[:,1]),2)
        else:
            return int("".join([str(int(x)) for x in array]),2)

    def i_to_b(self,integer):
        """
        take a state interger and convert it into a list of
        bits that has leading zeros to ensure the same length
        If a list is provided, then returns a 2d array of
        binary arrays.
        """
        if len(np.atleast_2d(integer)[0]) > 1:
            new_list = [self.i_to_b(x) for x in integer]
            return np.atleast_2d(new_list)
        else:
            integer = int(integer)
            new_list = [digit for digit 
                    in bin(integer)[2:]]
            for i in range(self.max_bit - len(new_list)):
                new_list.insert(0,0)
            return new_list

    def perturb_state(self,int_array,split_int,method):
        #apply a step to the state of the interface by either
        #switching one atom or choosing a random state..
        if method == 'multi-atom':
            new_state = random.randint(0,2**self.max_bit-1)
        else:
            step = random.randint(0,len(int_array)-1)
            int_array[step,1] = (int_array[step,1] + 1)%2
            new_state = self.b_to_i(int_array)
        if new_state in self.state_memory:
            if method != 'multi-atom':
                int_array[step,1] = (int_array[step,1] + 1)%2
            return int_array, split_int, False
        else:
            int_array, split_int = self.restore_state(
                            int_array,new_state,split_int)
            return int_array, split_int, True

    def print_results(self):

        result_file = self.input.dict['flip_file_name']+'.results.out'
        result_array = self.state_memory[self.state_memory[:,1].argsort()]

        try:
            os.remove(result_file)
        except:
            pass

        for i in result_array:
            printx("State = "+str(int(i[0]))+
                   " Energy = "+str(i[1]),result_file)

        return

    def restore_state(self,int_array,state,split_int):
        #given the current state array and the desired state
        #represented as an integer, return the structure
        #and state array in the new state
   
        new_state = ('{0:0'+str(len(int_array))+'b}').format(state)
        new_state = [int(x) for x in list(new_state)]

        for i in range(len(int_array)):
            if int_array[i,1] != new_state[i]:
                if int_array[i,1]:
                    int_array[i,1] = 0
                    split_int.atom[int_array[i,0]].position[2] -= (
                        float(self.input.dict['flip_separation']))
                else:
                    int_array[i,1] = 1
                    split_int.atom[int_array[i,0]].position[2] += (
                        float(self.input.dict['flip_separation']))

        return int_array,split_int

    def final_reductions(self,int_array,split_int):
        result_array = self.state_memory[self.state_memory[:,1].argsort()]
        printx("\nFinal reductions in lowest energy states",
                self.output_file)

        #loop over the n-lowest energy structures found in 
        for i in range(int(self.input.dict['flip_n_final_reduction'])):
            starting_state = int(result_array[i,0])
            lowest_energy = 0
            lowest_state = starting_state
            #set structure to starting state
            int_array,split_int = self.restore_state(
                    int_array,starting_state,split_int)
            bit_state = int_array[:,1].copy()
            
            #Loop over the number of passes
            for m in range(int(self.input.dict['flip_n_final_passes'])):
                #loop over all the bits
                for j in range(self.max_bit):
                    bit_state[j] = (bit_state[j] + 1) % 2
                    #check if we already have the energy for this state
                    if self.b_to_i(bit_state) not in self.state_memory:
                        int_array,split_int = self.restore_state(
                            int_array,self.b_to_i(bit_state),split_int)
                        try:
                            EnergyCalc(
                                self.input, self.input.dict['energy_method'],
                                split_int, 'run')
                        except Exception as err:
                            [printx(x) for x in err.args]
                        split_int.file_name = (
                                    self.input.dict['flip_file_name']+
                                    "."+str(self.b_to_i(bit_state)))
                        write_xyz(split_int)
                        self.state_memory = np.append(self.state_memory,
                            [[self.b_to_i(bit_state),split_int.energy]],0)
                        if split_int.energy < lowest_energy:
                            lowest_energy = split_int.energy
                            lowest_state = self.b_to_i(bit_state)
                        else:
                            bit_state[j] = (bit_state[j]+1)%2
                            int_array,split_int = self.restore_state(
                                int_array,self.b_to_i(bit_state),split_int)
                    else:
                        sm_temp = self.state_memory[np.argwhere(
                            self.state_memory == self.b_to_i(bit_state))[0,0],1]
                        if sm_temp < lowest_energy:
                            lowest_energy = sm_temp
                            lowest_state = self.b_to_i(bit_state)
                        else:
                            bit_state[j] = (bit_state[j]+1)%2
                            int_array,split_int = self.restore_state(
                                int_array,self.b_to_i(bit_state),split_int)

                    
            printx('State = '+str(lowest_state)+' Energy = '
                    +str(lowest_energy),self.output_file)
        return

    def gpr_state(self,int_array,atom):
        """
        Instantiates the gaussian process regressor and fits the data
        to it.
        """
        try:
            from .rbf_kernel import RBF_bin
            from sklearn.gaussian_process import GaussianProcessRegressor
        except Exception as err:
            raise Exception(
                "Error: couldn't import sklearn")
            sys.exit(1)

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        X_vals = self.i_to_b(self.state_memory[:,0])
        Y_vals = np.asarray(self.state_memory[:,1])
        kernel = RBF_bin(length_scale=1.0)
        
        gpr = GaussianProcessRegressor(kernel=kernel,alpha=1.0,
                    normalize_y=True)
        gpr.fit(X_vals,Y_vals)

        return gpr

    def expected_improvement(self, X, gpr, xi=0.01):
        """
        Calculate the expected improvement for an unknown set of
        states based on the state_memory.
        """
        X = np.rint(X)
        X = np.atleast_2d(X)
        mu, sigma = gpr.predict(X, return_std=True)
        if sigma == 0.0:
            raise Exception(
            """Sigma value has gone to 0 in GPR. Likely too many
            data points being used.  Set the total number of states
            equal to """+str(self.current_state+1)+""" and rerun
            to ensure the results file is generated.""")
            sys.exit(1)

        mu_sample = gpr.predict(self.i_to_b(self.state_memory[:,0]))

        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp*norm.cdf(Z)+sigma*norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def next_state(self,acq_func, gpr, n_restarts=25):
        """
        Chooses the next state by optimizing the combination
        of the expected improvement in the model and in
        decreasing the energy.
        """
        min_val = 1
        min_x = None

        def min_obj(X):
            return acq_func(X, gpr)

        bounds = []
        for i in range(self.max_bit):
            bounds.append([0,1])
        bounds = np.atleast_2d(bounds)

        for x0 in np.random.uniform(
                bounds[:,0],bounds[:,1],size=(n_restarts,self.max_bit)):
            res = minimize(min_obj, x0=x0, bounds = bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x
        
        return np.rint(min_x)


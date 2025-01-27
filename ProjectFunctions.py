import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
from sympy import *
import math
import qutip
import qutip as Q
import random
import json
import networkx as nx
from decimal import Decimal
import ast
import scipy.stats as stats
import sklearn as skl
from sklearn.cluster import SpectralClustering


def save_dict_to_file(dictionary, filename):
    #This function will save a dictionary as a text file
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")



def ising_model_hamiltonian(L, J, g, periodic): # L = length of spin chain, J = overall constant, g = constant applied to field term
    # Identity and Pauli matrices for spin 1/2
    I = Q.qeye(2)
    z = Q.sigmaz()
    x = Q.sigmax()

    # create hamiltonian
    H = 0

    if periodic == 0:
#         print("The string of spins is not periodic")
    
        # Interaction term
        for i in range(L - 1):
            # Tensor product betweean spin i and i+1. Other positions are identity
            term = Q.tensor([I] * i + [z] + [z] + [I] * (L - i - 2))
            H += term
            #print("The interaction term is:")
            #print(term)

        # Transverse field term
        for i in range(L):
            # Pauli x matrix at each spin in the chain while the rest are identity
            hterm = Q.tensor([I] * i + [x] + [I] * (L - i - 1))
            #print("The transverse field term is:")
            #print(hterm)
            H += g * hterm
            
    elif periodic == 1:
#         print("The string of spins is periodic")
        for i in range(L - 1):
            # Tensor product betweean spin i and i+1. Other positions are identity
            term = Q.tensor([I] * i + [z] + [z] + [I] * (L - i - 2))
            H += term
            #print("The interaction term is:")
            #print(term)
            
        periodicterm = Q.tensor([I] * 0 + [z] + [I] * (L-2) + [z])
        H += periodicterm
        
        # Transverse field term
        for i in range(L):
            # Pauli x matrix at each spin in the chain while the rest are identity
            hterm = Q.tensor([I] * i + [x] + [I] * (L - i - 1))
            H += g * hterm
        
        
    H = -J * H
#     print("The Hamiltonian is:")
    return H



def svd(a):
    print("the original matrix is")
    print(a)
    b = np.linalg.eig(np.matmul(np.transpose(a),a))
    #c = np.linalg.eig(np.matmul(a,np.transpose(a)))
    eigenvalues, eigenvectors = b
    
    #order eigenvalues and vectors in descending order
    idx = eigenvalues.argsort()[::-1] 
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    #set near zero terms equal to zero
    #print(eigenvalues)
    threshold1 = abs(eigenvalues) < 10**(-8)
    eigenvalues[threshold1] = 0
    
    #set near zero terms equal to zero
    #print(eigenvectors)
    threshold2 = abs(eigenvectors) < 10**(-8)
    eigenvectors[threshold2] = 0
    
    #trim zeroes from eigenvalues to make matrix sigma
    print("the eigenvalues are")
    print(eigenvalues)
    eigenvalstrimmed = np.trim_zeros(eigenvalues)
    print("the trimmed eigenvalues are")
    print(eigenvalstrimmed)
    
    #create matrix v
    print("the eigenvectors are")
    print(eigenvectors)
    v = np.transpose(eigenvectors)
    print("v is")
    print(v)
    
    #Find the singular values
    dimrow = np.count_nonzero(eigenvalstrimmed)
    dimcolumn = len(eigenvalues)
    sigma = np.zeros((dimrow,dimcolumn), dtype = float)
    svals = np.sqrt(eigenvalstrimmed)
    print("the singular values are")
    print(svals)
    
    #create matrix sigma
    np.fill_diagonal(sigma, svals)
    print("sigma is")
    print(sigma)
    print()
    
    #create matrix u
    u = np.zeros((a.shape[0], dimrow))
    for i in range(dimrow):
        u[:, i] = np.matmul(a/svals[i], eigenvectors[:, i])
    print("u is")
    print(u)
    print()
    
    #test if the SVD was correct
    originaltest = np.matmul(np.matmul(u,sigma), v)
    print("u*sigma*v gives")
    print(originaltest)
    print()
    
    #print the SVD
    print("the singular value decomposition for the matrix is")
    return u, sigma, v




def vec_to_matrix(spin_chain_length, vector, a_index, b_index):
    #function takes a length of spin chain, vector, and the index of particles in a or b
    #returns corresponding mapping to a matrix for the vector
    #only works for spin chain length = 3
    length = 2**spin_chain_length
    marray = []
    for i in range(length):
        binary = bin(i)[2:]
        long_binary = binary.zfill(spin_chain_length)
        array = [int(digit) for digit in long_binary]
        marray.append(array)
    #print(marray)
    
    if len(a_index) == 1: #if there is one spin in a
        a_positions = []
        for i in range(len(a_index)):
            pos = a_index[i]
            for j in range(len(marray)):
                a_positions.append(marray[j][pos])
                marray[j].pop(pos)
                b_positions = marray
        #print(a_positions)
        #print(b_positions)

        b_cols = []
    
        for i in b_positions:
            # Convert the list of 1s and 0s to a binary string
            binary_string = ''.join(map(str, i))
            # Convert the binary string to an integer
            number = int(binary_string, 2)
            b_cols.append(number)
    


        matrix = np.zeros((max(a_positions)+1, max(b_cols)+1), dtype=float)
    
        for value, row, col in zip(vector, a_positions, b_cols):
            matrix[row, col] = float(value[0])
        
#         print("The matrix is")
        return(matrix)
        
    
    elif len(b_index) == 1: #if there is one spin in b
        b_positions = []
        for i in range(len(b_index)):
            pos = b_index[i]
            for j in range(len(marray)):
                b_positions.append(marray[j][pos])
                marray[j].pop(pos)
                a_positions = marray
        #print(a_positions)
        #print(b_positions)

        a_rows = []
    
        for i in a_positions:
            # Convert the list of 1s and 0s to a binary string
            binary_string = ''.join(map(str, i))
            # Convert the binary string to an integer
            number = int(binary_string, 2)
            a_rows.append(number)
    
#         print("The corresponding rows are")
#         print(a_rows)
#         print("The corresponding columns are")
#         print(b_positions)
    
        matrix = np.zeros((max(a_rows)+1, max(b_positions)+1), dtype=float)
    
        for value, row, col in zip(vector, a_rows, b_positions):
            matrix[row, col] = float(value[0])
        
#         print("The matrix is")
        return matrix



def mutual_info(length, psi, A, B):
#length = length of spin chain
#psi = groundstate vector from respective model
#A = A spin
#B = B spin

#create a selection of "everything else" based on given A and B
    C = [item for i, item in enumerate(range(length)) if i not in A and i not in B]
    
#create reduced density matrices for each partition
    rhoA = Q.ptrace(psi,A)
    rhoB = Q.ptrace(psi,B)
    rhoC = Q.ptrace(psi,C)
    
#compute eigenvalues and entropy for A
    eigenvalsA = rhoA.eigenenergies()
    threshold1 = abs(eigenvalsA) < 10**(-8)
    eigenvalsA[threshold1] = 0
    #print(eigenvals)
    eigenvalscleanedA = [num for num in eigenvalsA if num != 0]
    #print(eigenvalscleaned)
    vnentropylistA = []
    for l in eigenvalscleanedA:
        vnentropylistA.append(-l*np.log(l))
        #print(vnentropylist)
        vnentropyA = sum(vnentropylistA)
    #print(vnentropyA)
    
#compute eigenvalues and entropy for B
    eigenvalsB = rhoB.eigenenergies()
    threshold1 = abs(eigenvalsB) < 10**(-8)
    eigenvalsB[threshold1] = 0
    #print(eigenvals)
    eigenvalscleanedB = [num for num in eigenvalsB if num != 0]
    #print(eigenvalscleaned)
    vnentropylistB = []
    for l in eigenvalscleanedB:
        vnentropylistB.append(-l*np.log(l))
        #print(vnentropylist)
        vnentropyB = sum(vnentropylistB)
    #print(vnentropyB)
    
#compute eigenvalues and entropy for C
    eigenvalsC = rhoC.eigenenergies()
    threshold1 = abs(eigenvalsC) < 10**(-8)
    eigenvalsC[threshold1] = 0
    #print(eigenvals)
    eigenvalscleanedC = [num for num in eigenvalsC if num != 0]
    #print(eigenvalscleaned)
    vnentropylistC = []
    for l in eigenvalscleanedC:
        vnentropylistC.append(-l*np.log(l))
        #print(vnentropylist)
        vnentropyC = sum(vnentropylistC)
    #print(vnentropyC)
    
#compute the mutual information based on the previously computed entropy
    mutualinfo = vnentropyA + vnentropyB - vnentropyC
#compute distance proxy from mutual information
    distance = (-np.log(mutualinfo/(2*np.log(2))))
    #distance = (-np.log(mutualinfo))/(2*np.log(2))
    return mutualinfo, distance#, vnentropyA, vnentropyB, vnentropyC




def correlation_function(length, psi, zvar):
#length = length of spin chain
#psi = groundstate from respective model
#zvar determines if it is the z or x corellation function 0 = z, 1 = x
    if zvar == 0:
        I = Q.qeye(2)
        z = Q.sigmaz()
    elif zvar == 1:
        I = Q.qeye(2)
        z = Q.sigmax()
    
    if zvar == 0:
        xorz = 'z'
    elif zvar == 1:
        xorz = 'x'
    
    combinations = []
    physdist = []
# Generate all possible pairs of positions
    for i in range(length):
        for j in range(i + 1, length):
            combinations.append([i, j])
            physdist.append(j-i)
    #print(combinations)

    czz1terms = []
    czz2terms = []

# Iterate through each combination
    for combo in combinations:
        tensors1 = [I for _ in range(length)]
# Construct the tensor product for this combination and then the expectation value
        for i in combo:
            tensors1[i] = z
        term = Q.tensor(tensors1)
        expecvalue = Q.expect(term, psi)
        czz1terms.append(expecvalue)
    #print(czz1terms)
    
#constructing tensor product for each position in chain length with z, expectation value of each, then product of these
    for combo in combinations:
        tensors2 = [I for _ in range(length)]
        expecvalues = []
        for i in combo:
            tensors2[i] = z
            term = Q.tensor(tensors2)
            expecvalue = Q.expect(term, psi)
            expecvalues.append(expecvalue)
        termfinal = expecvalues[0] * expecvalues[1]
        czz2terms.append(termfinal)
    #print(czz2terms)
    
#compute correlation function from its two parts    
    correlationfuncvals = []
    for i in range(len(czz1terms)):
        cval = czz1terms[i] - czz2terms[i]
        correlationfuncvals.append(cval)

#compute distance from the correlation function values        
    distances = []
    for i in correlationfuncvals:
        distances.append(-np.log(i))
    
    #return combinations, physdist, czz1terms, czz2terms,
    return correlationfuncvals, distances




def correlation_function_one_pos(length, pos, psi, zvar):
#length = length of spin chain
#pos = position in spin chain with which to make the possible combinations from
#psi = groundstate from respective model
#zvar determines if it is the z or x corellation f unction 0 = z, 1 = x
    if zvar == 0:
        I = Q.qeye(2)
        z = Q.sigmaz()
    elif zvar == 1:
        I = Q.qeye(2)
        z = Q.sigmax()
    
    combinations = []
    physdist = []
# Generate all possible pairs of positions
    for i in range(length):
        if i != pos:
            combinations.append([range(length)[pos], range(length)[i]])
            dist = np.abs(range(length)[pos] - range(length)[i])
            physdist.append(dist)

    czz1terms = []
    czz2terms = []

# Iterate through each combination
    for combo in combinations:
        tensors1 = [I for _ in range(length)]
# Construct the tensor product for this combination and then the expectation value
        for i in combo:
            tensors1[i] = z
        term = Q.tensor(tensors1)
        expecvalue = Q.expect(term, psi)
        czz1terms.append(expecvalue)
    #print(czz1terms)
    
#constructing tensor product for each position in chain length with z, expectation value of each, then product of these
    for combo in combinations:
        tensors2 = [I for _ in range(length)]
        expecvalues = []
        for i in combo:
            tensors2[i] = z
            term = Q.tensor(tensors2)
            expecvalue = Q.expect(term, psi)
            expecvalues.append(expecvalue)
        termfinal = expecvalues[0] * expecvalues[1]
        czz2terms.append(termfinal)
    #print(czz2terms)
    
#compute correlation function from its two parts    
    correlationfuncvals = []
    for i in range(len(czz1terms)):
        cval = czz1terms[i] - czz2terms[i]
        correlationfuncvals.append(cval)

#compute distance from the correlation function values        
    distances = []
    for i in correlationfuncvals:
        distances.append(-np.log(np.abs(i)))
        
    
    #return combinations, physdist, czz1terms, czz2terms,
    return correlationfuncvals, distances


def excited_states(eigenenergies, minenergy, degeneracy):
    #This function will take a list of eigenenergies and compare them all to the ground state energy.
    #It returns a number of excited state energies equal to the degeneracy of the system as well as the energy difference to the ground state
    excitedstates = []
    seen_states = set()
    energy_diff = []
    
    for energy in eigenenergies:
        if abs(energy - minenergy) > 10e-12 and energy not in seen_states:
            excitedstates.append(energy)
            seen_states.add(energy)
            if len(excitedstates) == degeneracy:
                break
    for energy in excitedstates:
        diff = energy - minenergy
        energy_diff.append(diff)
                
    return excitedstates, energy_diff



def simplified_groundstate(length, states):
    #this, given the size of the system and its classical ground states written in binary will turn each binary groundstate into a vector in qutip
    #it can then be used to create a classical superposition of groundstates for analysis
    statevecs = []
    tensors = ['' for _ in range(length)]
    for state in states:
        #print(state)
        for i in range(length):
            index = state[i]
            #print(index)
            if index == '1':
                tensors[i] = Q.basis(2,1)
            elif index == '0':
                tensors[i] = Q.basis(2,0)
        #print(tensors)
        statevecs.append(Q.tensor(tensors))
        
    return statevecs

def find_degeneracy(length, Hintdiag):
    #this function takes the size of the system and the  diagonal of the interaction term of the hamiltonian
    #it returns the diagonal, its classical ground state energy, degeneracy of the classical ground state, position of each ground state energy in the diagonal, and corresponding classical ground states written in binary
    minval = min(Hintdiag)
    minenergy = minval.real
    #degeneracy = Hintdiag.count(minenergy)
    degeneracy = np.count_nonzero(Hintdiag == minenergy)
    groundpos = []
    for index, value in enumerate(Hintdiag):
        if np.abs(value - minenergy) < 10**(-12):
            #print(np.abs(value - minenergy))
            groundpos.append(index)
            
    states = []
    for index in groundpos:
        states.append(bin(index)[2:].zfill(length))
            
    return minenergy, degeneracy, groundpos, states


def groundstate_prob(length, jij, hlist):
    #This takes the size of the system, the interaction term array, and a list of h values
    #It will return 
    probsoverh = {}
    probs1 = []
    probs2 = []
    probs3 = []
    probs4 = []
#     probs5 = []
#     probs6 = []
#     probs7 = []
#     probs8 = []
    Hint, Hintdiag = sg_interaction(length, jij)
    Hfield = sg_field(length)
    for h in hlist:
        H = Hint + (h * Hfield)
        E, psi = H.groundstate()
        minenergy, degeneracy, groundpos, states = find_degeneracy(length, Hintdiag)
        probperground = {}
        for i in range(len(groundpos)):
            pos = groundpos[i]
            probperground.update({states[i]: np.real(psi[pos][0]*np.conj(psi[pos][0]))})
        probsoverh.update({h:probperground})
        probs1.append(probperground[states[0]])
        probs2.append(probperground[states[1]])
        probs3.append(probperground[states[2]])
        probs4.append(probperground[states[3]])
#         probs5.append(probperground[states[4]])
#         probs6.append(probperground[states[5]])
#         probs7.append(probperground[states[6]])
#         probs8.append(probperground[states[7]])
    return probsoverh, probs1, probs2, probs3, probs4#, probs5, probs6, probs7, probs8, states




def hamming_distance(length, Hintdiag):
    minenergy, degeneracy, groundpos, states = find_degeneracy(length, Hintdiag)
    for i in range(len(groundpos)):
        for j in range(i + 1, len(groundpos)):
            if bin(groundpos[i] ^ groundpos[j]).count('1') < length:
                print('The hamming distance is ' + str(bin(groundpos[i] ^ groundpos[j]).count('1')) + ' for states ' + states[i] + ' and ' + states[j]) 




def generate_jij(length, num_ones, seed):
    jijlength = length*(length - 1)/2
    num_neg_ones = jijlength - num_ones
    # Create an array with the specified number of 1s and -1s
    jij = np.array([1] * int(num_ones) + [-1] * int(num_neg_ones))
    
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Shuffle the array to randomize the order
    np.random.shuffle(jij)
    
    return jij




def spin_glass_hamiltonian(length, jij, h): 
# length = length of spin chain,
#jij = random list of j values to be multiplied by each term
#h = constant applied to field term

# Identity and Pauli matrices for spin 1/2
    I = Q.qeye(2)
    z = Q.sigmaz()
    x = Q.sigmax()

# create hamiltonian
    H = 0
    
    combinations = []
# Generate all possible pairs of positions
    for i in range(length):
        for j in range(i + 1, length):
            combinations.append([i, j])
    #print(combinations)

    terms = []
    
    
# Iterate through each combination
    Hintpos = 0
    for i, combo in enumerate(combinations):
        #print(i,combo)
        j = jij[i]
        #print(j)
        tensors = [I for _ in range(length)]
# Construct the tensor product for this combination
        for k in combo:
            tensors[k] = z
        term = Q.tensor(tensors)
        terms.append(term)
        
# multiply by jij
        #print(j)
        Hintpos += j * term
        Hint = -Hintpos
        Hintdiag = Hint.diag()
        
    #print(terms)

# Transverse field term
    Hfieldpos = 0
    for v in range(length):
# Pauli x matrix at each spin in the chain while the rest are identity
        hterm = Q.tensor([I] * v + [x] + [I] * (length - v - 1))
            #print("The transverse field term is:")
            #print(hterm)
        Hfieldpos += hterm
        Hfield = -Hfieldpos
#     print("The Hamiltonian is:")

        H = Hint + (h*Hfield)

    return H, Hintdiag



def sg_interaction(length, jij): 
# length = length of spin chain,
#jij = random list of j values to be multiplied by each term
#h = constant applied to field term

# Identity and Pauli matrices for spin 1/2
    I = Q.qeye(2)
    z = Q.sigmaz()
    x = Q.sigmax()

# create hamiltonian
    Hintpos = 0
    
    combinations = []
# Generate all possible pairs of positions
    for i in range(length):
        for j in range(i + 1, length):
            combinations.append([i, j])
    #print(combinations)

    terms = []
    
    
# Iterate through each combination
    for i, combo in enumerate(combinations):
        #print(i,combo)
        j = jij[i]
        #print(j)
        tensors = [I for _ in range(length)]
# Construct the tensor product for this combination
        for k in combo:
            tensors[k] = z
        term = Q.tensor(tensors)
        terms.append(term)
        
# multiply by jij
        #print(j)
        Hintpos += j * term
        Hint = -Hintpos
        Hintdiag = Hint.diag()
        
    return Hint, Hintdiag




def sg_field(length): 
# length = length of spin chain,
#h = constant applied to field term
    I = Q.qeye(2)
    z = Q.sigmaz()
    x = Q.sigmax()

    Hfieldpos = 0
# Transverse field term
    for v in range(length):
# Pauli x matrix at each spin in the chain while the rest are identity
        hterm = Q.tensor([I] * v + [x] + [I] * (length - v - 1))
            #print("The transverse field term is:")
            #print(hterm)
        Hfieldpos += hterm
        Hfield = -Hfieldpos
#     print("The Hamiltonian is:")

    return Hfield



def sort_seeds(length, num_ones, seedlist):
    seeddegeneracy = []
    degeneracylist = []
    seeddict = {}
    for i in seedlist:
        jij = generate_jij(length, num_ones, i)
        H, Hintdiag = spin_glass_hamiltonian(length, jij, 0)
        minenergy, degeneracy, groundpos, states = find_degeneracy(length, Hintdiag)
        seeddict.update({i:[minenergy, degeneracy, states]})
    sorted_seeds = {k: seeddict[k] for k in sorted(seeddict, key=lambda k: seeddict[k][1], reverse=True)}
#         degeneracylist.append(degeneracy)
#         seeddegeneracy.append([i,degeneracy])
#         sortedseeds = sorted(seeddegeneracy, key=lambda x: x[1], reverse=True)
#     maxdegen = max(degeneracylist)
#     seeds = [seed for seed, degeneracy in enumerate(degeneracylist) if degeneracy == maxdegen]
#     return maxdegen, sortedseeds, seeds
    return sorted_seeds



def max_variance_ratio(distancedict, hlist):
    ratiodict = {}
    stdevdict = {}
    meandict = {}
    for h in hlist:
        distances = distancedict[h]
        standarddev = np.std(distances)
        mean = np.mean(distances)
        ratio = standarddev / mean
        ratiodict.update({h:ratio})
        stdevdict.update({h:standarddev})
        meandict.update({h:mean})
    maxh = max(ratiodict, key=ratiodict.get)
    maxratio = ratiodict[maxh]
    #print('The hval with highest ratio is ' + str (maxh) + ' with ratio ' + str(maxratio))
    return ratiodict, maxh, maxratio, stdevdict, meandict


def find_transitions(xaxis, stagmagvals, ratiodict, stdevdict, meandict):
    
    covarvals = []
    for key in ratiodict.keys():
        covarvals.append(ratiodict[key])
    
    stdevvals = []
    for key in stdevdict.keys():
        stdevvals.append(stdevdict[key])
    
    meanvals = []
    for key in meandict.keys():
        meanvals.append(meandict[key])
    
    
    # Calculate numerical derivatives (slope between adjacent points)
    dx = np.diff(xaxis)
    dstagmag = np.diff(stagmagvals)
    slopes1 = dstagmag / dx

    # Find the index of the maximum slope
    stagmag_slope_index = np.argmax(np.abs(slopes1))

    # Get the x and y values of the point with the maximum slope
    x_max_slope1 = xaxis[stagmag_slope_index + 1]  # +1 because slopes are between points
    stagmag_max_slope = stagmagvals[stagmag_slope_index + 1]

    #print(f"Point of maximum slope for staggered magnetization: ({x_max_slope1}, {stagmag_max_slope})")
    
    # Calculate numerical derivatives (slope between adjacent points)
    
    dx = np.diff(xaxis)
    dmean = np.diff(meanvals)
    slopes2 = dmean / dx

    # Find the index of the maximum slope
    mean_slope_index = np.argmax(np.abs(slopes2))

    # Get the x and y values of the point with the maximum slope
    x_max_slope2 = xaxis[mean_slope_index + 1]  # +1 because slopes are between points
    mean_max_slope = meanvals[mean_slope_index + 1]

    #print(f"Point of maximum slope for mean: ({x_max_slope2}, {mean_max_slope})")
    
    
    # Use scipy.signal.find_peaks to find the indices of the peaks
    peaks1, _ = scipy.signal.find_peaks(stdevvals)
    #print(peaks)

    # Print the x and y coordinates of the peaks
    peak_x1 = xaxis[peaks1[0]]
    peak_stdev = stdevvals[peaks1[0]]
    #print(peak_x1, peak_stdev)
    #print(f"Peak for standard deviation: ({peak_x1}, {peak_stdev})")
    
    
    
    # Use scipy.signal.find_peaks to find the indices of the peaks
    peaks2, _ = scipy.signal.find_peaks(covarvals)
    #print(peaks)

    # Print the x and y coordinates of the peaks
    peak_x2 = xaxis[peaks2[0]]
    peak_covar = covarvals[peaks2[0]]
    #print(peak_x1, peak_covar)
    #print(f"Peak for coefficient of variance: ({peak_x2}, {peak_covar})")
    
    return x_max_slope1, peak_x2, x_max_slope2, peak_x1



# def find_transitions_low_h(xaxis, stagmagvals, ratiodict, stdevdict, meandict):
#     
#     covarvals = []
#     for key in ratiodict.keys():
#         covarvals.append(ratiodict[key])
#     
#     stdevvals = []
#     for key in stdevdict.keys():
#         stdevvals.append(stdevdict[key])
#     
#     meanvals = []
#     for key in meandict.keys():
#         meanvals.append(meandict[key])
#     
#     
#     # Calculate numerical derivatives (slope between adjacent points)
#     dx = np.diff(xaxis)
#     dstagmag = np.diff(stagmagvals)
#     slopes1 = dstagmag / dx
# 
#     # Find the index of the maximum slope
#     stagmag_slope_index = np.argmax(np.abs(slopes1))
# 
#     # Get the x and y values of the point with the maximum slope
#     x_max_slope1 = xaxis[stagmag_slope_index + 1]  # +1 because slopes are between points
#     stagmag_max_slope = stagmagvals[stagmag_slope_index + 1]
# 
#     #print(f"Point of maximum slope for staggered magnetization: ({x_max_slope1}, {stagmag_max_slope})")
#     
#     # Calculate numerical derivatives (slope between adjacent points)
#     xaxisnew = xaxis[-59:]
#     
#     dx = np.diff(xaxisnew)
#     dmean = np.diff(meanvals[-59:])
#     slopes2 = dmean / dx
# 
#     # Find the index of the maximum slope
#     mean_slope_index = np.argmax(np.abs(slopes2))
# 
#     # Get the x and y values of the point with the maximum slope
#     x_max_slope2 = xaxisnew[mean_slope_index + 1]  # +1 because slopes are between points
#     mean_max_slope = meanvals[mean_slope_index + 1]
# 
#     #print(f"Point of maximum slope for mean: ({x_max_slope2}, {mean_max_slope})")
#     
#     
#     # Use scipy.signal.find_peaks to find the indices of the peaks
#     peaks1, _ = scipy.signal.find_peaks(stdevvals)
#     #print(peaks)
# 
#     # Print the x and y coordinates of the peaks
#     peak_x1 = xaxis[peaks1[0]]
#     peak_stdev = stdevvals[peaks1[0]]
#     #print(peak_x1, peak_stdev)
#     #print(f"Peak for standard deviation: ({peak_x1}, {peak_stdev})")
#     
#     
#     
#     # Use scipy.signal.find_peaks to find the indices of the peaks
#     peaks2, _ = scipy.signal.find_peaks(covarvals)
#     #print(peaks)
# 
#     # Print the x and y coordinates of the peaks
#     peak_x2 = xaxis[peaks2[0]]
#     peak_covar = covarvals[peaks2[0]]
#     #print(peak_x1, peak_covar)
#     #print(f"Peak for coefficient of variance: ({peak_x2}, {peak_covar})")
#     
#     return x_max_slope1, peak_x2, x_max_slope2, peak_x1



def staggered_magnetization(length, states):#, psi):
    #will calculate the staggered magnetization opertor
    #only works for degeneracy 2
    statesigns = []
    z = Q.sigmaz()
    I = Q.qeye(2)
    for spin in states[0]:
        if spin == '0':
            sign = 1
            statesigns.append(sign)
        elif spin == '1':
            sign = -1
            statesigns.append(sign)
            
            
            
    zterm = 0
    for i in range(length):
        # Pauli z matrix at each spin in the chain while the rest are identity
        term = Q.tensor([I] * i + [z] + [I] * (length - i - 1))
        #zterm += (staggersign * term)
        zterm += (statesigns[i] * term)
    stagmag = (1/length) * zterm
    #expectvalue = Q.expect((stagmag**2), psi)
    return stagmag#, expectvalue





def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")
            
            
def save_dict_to_array_file(dictionary, filename):
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            line = f"{key} {value}\n"
            file.write(f"{key} {value}\n")
        #print(array)
            
            
            
def generate_sorted_classical_data(length, num_ones, seedlist):
    seeddict = {}
    for seed in seedlist:
        jij = generate_jij(length, num_ones, seed)
        Hint, Hintdiag = sg_interaction(length, jij)
        minenergy, degeneracy, groundpos, states = find_degeneracy(length, Hintdiag)
        seeddict.update({seed:[degeneracy, minenergy, states]})
    sorted_seeds = {k: seeddict[k] for k in sorted(seeddict, key=lambda k: seeddict[k], reverse=True)}
    return sorted_seeds


def read_from_disk(filename):
    #works for the save dict to array function and will split into an array containing two parts which were separated by the first space in the text file
    info = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split(' ', 1)
            info.append(parts[0])
            info.append(ast.literal_eval(parts[1].strip()))
        return info
    

def read_array_data(filename):
# Open and read the file
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Remove the newline character and any leading/trailing spaces
            line = line.strip()
            
            # Split the line into the first number and the list part
            parts = line.split(' ', 1)  # Split into two parts at the first space
        
            # The first part is the first number, and the second part is the list
            first_number = float(parts[0])  # Convert the first number to float
            list_numbers = eval(parts[1])  # Convert the string representation of the list to an actual list
        
            # Combine the first number and the rest of the numbers
            row = [first_number] + list_numbers
        
            # Append the row to the data list
            data.append(row)

    # Convert the data list to a numpy array
    array = np.array(data)

    # Print the numpy array
    return array


def read_classical_data(filename):
    # Create an empty list to store the rows
    data = []

    # Open and read the file
    with open(filename, 'r') as file:
        for line in file:
            # Remove the newline character and any leading/trailing spaces
            line = line.strip()

            # Split the line into the first number, the second number, and the binary string list
            parts = line.split(' ', 1)  # Split into two parts at the first space

            # The first part is the first number, and the second part is the rest (the list part)
            first_number = parts[0]  # Convert the first number to float
            second_part = eval(parts[1])  # Use eval to turn the string into an actual list

            # Extract the second number and the list of binary strings
            second_number = second_part[0]# The second number is the first element in the list
            third_number = second_part[1]
            binary_list = second_part[2]  # The list of binary strings

            # Combine the first number, second number, and the binary list
            row = [first_number, second_number, third_number, binary_list]

            # Append the row to the data list
            data.append(row)

    # Convert the data list to a numpy array
    array = np.array(data, dtype=object)

    # Print the numpy array
    return array
    
    
def generate_psi_data(length, num_ones, seedlist, Hfield, hval):
    seeddict = {}
    for seed in seedlist:
        jij = generate_jij(length, num_ones, seed)
        Hint, Hintdiag = sg_interaction(length, jij)
        H = Hint + (hval * Hfield)
        E, psi = H.groundstate()
        seeddict.update({seed:psi})
    return seeddict



def generate_mutual_information_data(length, num_ones, seedlist, Hfield, hval):
    seeddict = {}
    for seed in seedlist:
        jij = generate_jij(length, num_ones, seed)
        Hint, Hintdiag = sg_interaction(length, jij)
        H =  Hint + (hval * Hfield)
        E, psi = H.groundstate()
        minfo1 = []
        for i in range(length):
            for j in range(i + 1, length):
                mutualinfo, distance= mutual_info(length, psi, [i], [j])
                minfo1.append(mutualinfo)
        seeddict.update({seed:minfo1})
    return seeddict


def generate_transitions_data(length, Hfield, seedlist, hvals):
#     stagmag_transitions = {}
#     covar_transitions = {}
#     mean_transitions = {}
#     stdev_transitions = {}
    transitionsdict = {}


    for seed in seedlist:
        stagmaglist = []
        jij = generate_jij(length, 14, seed)
        Hint, Hintdiag = sg_interaction(length, jij)
        minenergy, degeneracy, groundpos, states = find_degeneracy(length, Hintdiag)
        stagmag = staggered_magnetization(length, states)
        for h in hvals:
            H = Hint + (h * Hfield)
            E, psi = H.groundstate()
            expectvalue = Q.expect((stagmag**2), psi)
            stagmaglist.append(expectvalue)
    

        alldist = {}
        allminfo = {}
        # Generate all possible pairs of positions
        for h in hvals:
            distancelist = []
            minfolist = []
            H = Hint + (h * Hfield)
            E, psi = H.groundstate()
            for i in range(length):
                for j in range(i + 1, length):
                    minfo, distance = mutual_info(length, psi, [i], [j])
                    distancelist.append(distance)
                    minfolist.append(minfo)
            alldist.update({h:distancelist})
            #allminfo.update({h:minfolist})
        #print(alldist)
        ratiodict, maxh, maxratio, stdevdict, meandict = max_variance_ratio(alldist, hvals)


        stagmag_max_slope, peak_covar, mean_max_slope, peak_stdev = find_transitions(hvals, stagmaglist, ratiodict, stdevdict, meandict)
    
#         stagmag_transitions.update({seed:stagmag_max_slope})
#         covar_transitions.update({seed:peak_covar})
#         mean_transitions.update({seed:mean_max_slope})
#         stdev_transitions.update({seed:peak_stdev})
        transitionsdict.update({seed:[stagmag_max_slope, mean_max_slope, peak_covar, peak_stdev]})
    return transitionsdict


def mutual_info_fast(length, psi, a, b):
#length = length of spin chain
#psi = groundstate vector from respective model
#A = A spin
#B = B spin

    A = [a]
    B = [b]

#create a selection of "everything else" based on given A and B
    AB = [a,b]
    
#create reduced density matrices for each partition
    rhoA = Q.ptrace(psi,A)
    rhoB = Q.ptrace(psi,B)
    rhoAB = Q.ptrace(psi,AB)
    
#compute eigenvalues and entropy for A
    eigenvalsA = rhoA.eigenenergies()
    threshold1 = abs(eigenvalsA) < 10**(-8)
    eigenvalsA[threshold1] = 0
    #print(eigenvals)
    eigenvalscleanedA = [num for num in eigenvalsA if num != 0]
    #print(eigenvalscleaned)
    vnentropylistA = []
    for l in eigenvalscleanedA:
        vnentropylistA.append(-l*np.log(l))
        #print(vnentropylist)
        vnentropyA = sum(vnentropylistA)
    #print(vnentropyA)
    
#compute eigenvalues and entropy for B
    eigenvalsB = rhoB.eigenenergies()
    threshold1 = abs(eigenvalsB) < 10**(-8)
    eigenvalsB[threshold1] = 0
    #print(eigenvals)
    eigenvalscleanedB = [num for num in eigenvalsB if num != 0]
    #print(eigenvalscleaned)
    vnentropylistB = []
    for l in eigenvalscleanedB:
        vnentropylistB.append(-l*np.log(l))
        #print(vnentropylist)
        vnentropyB = sum(vnentropylistB)
    #print(vnentropyB)
    
#compute eigenvalues and entropy for C
    eigenvalsAB = rhoAB.eigenenergies()
    threshold1 = abs(eigenvalsAB) < 10**(-8)
    eigenvalsAB[threshold1] = 0
    #print(eigenvals)
    eigenvalscleanedAB = [num for num in eigenvalsAB if num != 0]
    #print(eigenvalscleaned)
    vnentropylistAB = []
    for l in eigenvalscleanedAB:
        vnentropylistAB.append(-l*np.log(l))
        #print(vnentropylist)
        vnentropyAB = sum(vnentropylistAB)
    #print(vnentropyC)
    
#compute the mutual information based on the previously computed entropy
    mutualinfo = vnentropyA + vnentropyB - vnentropyAB
#compute distance proxy from mutual information
    distance = (-np.log(mutualinfo/(2*np.log(2))))
    #distance = (-np.log(mutualinfo))/(2*np.log(2))
    return mutualinfo, distance#, vnentropyA, vnentropyB, vnentropyC


def create_affinity_matrix(length, combinations, minfolist):
    #affmat = np.zeros((length,length))
    affmat = np.identity(length)
    for [i, j], minfo in zip(combinations, minfolist):
        affmat[i,j] = minfo/(2*np.log(2))
    for [n, m], minfo in zip(combinations, minfolist):
        affmat[m,n] = minfo/(2*np.log(2))
    return affmat


def affmat_analysis(affmat):
    #find eigenvalues
    eigenvals, vecs = scipy.linalg.eig(affmat)
    realeigenvals = np.real(eigenvals)
    gaps = []
    #find the gaps between each adjacent eigenvalue
    for i in range(1, len(realeigenvals)):
        gaps.append(abs(realeigenvals[i] - realeigenvals[i-1]))
        #print(abs(realeigenvals[i] - realeigenvals[i-1]))
    #find maximum gap
    max_gap = max(gaps)
    #convert to number of clusters
    #Will always be plus 1 to account for eigenvalue in index 0, as eigenvalues are always sorted max to min
    num_clusters = gaps.index(max_gap) + 1
    return num_clusters, max_gap


def graph_conductance(affmat, clusters):
    S_spins = []
    T_spins = []
    for index, cluster in enumerate(clusters):
        if cluster == 0:
            S_spins.append(index)
        if cluster == 1:
            T_spins.append(index)
            
    inter_cluster_weights = []
    for i in S_spins:
        for j in T_spins:
            inter_cluster_weights.append(affmat[i,j])
            
    inter_cluster_sum = sum(inter_cluster_weights)
    
    all_s_weights = []
    combinations = []
    for l in S_spins:
        for m in range(len(clusters)):
            if l != m and [m,l] not in combinations:
                combinations.append([l,m])
                #print(l,m)
                all_s_weights.append(affmat[l,m])
    all_s_weight_sum = sum(all_s_weights)
    
    
    conductance = inter_cluster_sum / all_s_weight_sum
    
        
    return conductance #, inter_cluster_sum, all_s_weight_sum



def classical_data_by_line(length, num_ones, seedlist, filename):
    with open(filename, 'a') as file:
        for seed in seedlist:
            jij = generate_jij(length, num_ones, seed)
            Hint, Hintdiag = sg_interaction(length, jij)
            minenergy, degeneracy, groundpos, states = find_degeneracy(length, Hintdiag)
            file.write(f"{seed} {[degeneracy, minenergy, states]}\n")
            
            
            
def generate_mutual_info_by_line(length, num_ones, seedlist, Hfield, hval, filename):
    with open(filename, 'a') as file:
        for seed in seedlist:
            jij = generate_jij(length, num_ones, seed)
            Hint, Hintdiag = sg_interaction(length, jij)
            H =  Hint + (hval * Hfield)
            E, psi = H.groundstate()
            minfo = []
            for i in range(length):
                for j in range(i + 1, length):
                    mutualinfo, distance = mutual_info(length, psi, [i], [j])
                    minfo.append(mutualinfo)
            file.write(f"{seed} {minfo}\n")
            
            
def read_from_seed_file(filename):
    seedlist = []
    with open(filename, 'r') as file:
        for line in file:
            seed = line.strip()
            seedlist.append(int(seed))
        return seedlist
    
    


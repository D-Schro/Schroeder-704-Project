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


def save_dict_to_file(dictionary, filename):
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



def create_bipartition(vec, a_size): #input vector as well as number of spins wanted in a. Remainder will be in partition b
    length = len(vec)
    indexlist = list(range(length))
    
    random.shuffle(indexlist) #create random list to ensure random selection
    
    # Select a subset of the list
    a_index = indexlist[:a_size]
    
    # The remaining b list
    b_index = indexlist[a_size:]

    print('The index for a spins is')
    print(a_index)
    print('The index for b spins is')
    print(b_index)



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
    
    combinations = []
    physdist = []
    # Generate all possible pairs of positions
    for i in range(length):
        for j in range(i + 1, length):
            combinations.append([i, j])
            physdist.append(j-i)
               
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
    return combinations, physdist, mutualinfo, distance#, vnentropyA, vnentropyB, vnentropyC




def correlation_function(length, psi, zvar):
#length = length of spin chain
#psi = groundstate from respective model
#zvar determines if it is the zor x corellation function 0 = z, 1 = x
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
    
    return combinations, physdist, czz1terms, czz2terms, correlationfuncvals, distances




def correlation_function_one_pos(length, pos, psi, zvar):
#length = length of spin chain
#pos = position in spin change with which to make the possible combinations from
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
        
    
    return combinations, physdist, czz1terms, czz2terms, correlationfuncvals, distances


def excited_states(eigenenergies, minenergy, degeneracy):
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
            
    return Hintdiag, minenergy, degeneracy, groundpos, states


def groundstate_prob(length, jij, hlist):
    probsoverh = {}
    probs1 = []
    probs2 = []
    probs3 = []
    probs4 = []
    for h in hlist:
        H, Hintdiag = spin_glass_hamiltonian(length, jij, h)
        E, psi = H.groundstate()
        Hintdiag, minenergy, degeneracy, groundpos, states = find_degeneracy(length, Hintdiag)
        probperground = {}
        for i in range(len(groundpos)):
            pos = groundpos[i]
            probperground.update({states[i]: np.real(psi[pos][0]*np.conj(psi[pos][0]))})
        probsoverh.update({h:probperground})
        probs1.append(probperground[states[0]])
        probs2.append(probperground[states[1]])
        probs3.append(probperground[states[2]])
        probs4.append(probperground[states[3]])
    return probsoverh, probs1, probs2, probs3, probs4, states




def hamming_distance(length, Hintdiag):
    Hintdiag, minenergy, degeneracy, groundpos, states = find_degeneracy(length, Hintdiag)
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
    Hint = 0
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
        Hint += j * term
        Hintdiag = Hint.diag()
        
    #print(terms)

# Transverse field term
    Hfield = 0
    for v in range(length):
# Pauli x matrix at each spin in the chain while the rest are identity
        hterm = Q.tensor([I] * v + [x] + [I] * (length - v - 1))
            #print("The transverse field term is:")
            #print(hterm)
        Hfield += hterm
#     print("The Hamiltonian is:")

        H = -1 * (Hint + (h*Hfield))

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

    Hfield = 0
# Transverse field term
    for v in range(length):
# Pauli x matrix at each spin in the chain while the rest are identity
        hterm = Q.tensor([I] * v + [x] + [I] * (length - v - 1))
            #print("The transverse field term is:")
            #print(hterm)
        Hfield += hterm
#     print("The Hamiltonian is:")

    return Hfield



def sort_seeds(length, num_ones, seedlist):
    seeddegeneracy = []
    degeneracylist = []
    seeddict = {}
    for i in seedlist:
        jij = generate_jij(length, num_ones, i)
        H, Hintdiag = spin_glass_hamiltonian(length, jij, 0)
        Hintdiag, minenergy, degeneracy, groundpos, states = find_degeneracy(length, Hintdiag)
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
    for h in hlist:
        distances = distancedict[h]
        standarddev = np.std(distances)
        mean = np.mean(distances)
        ratio = standarddev / mean
        ratiodict.update({h:ratio})
    maxh = max(ratiodict, key=ratiodict.get)
    maxratio = ratiodict[maxh]
    #print('The hval with highest ratio is ' + str (maxh) + ' with ratio ' + str(maxratio))
    return ratiodict, maxh, maxratio





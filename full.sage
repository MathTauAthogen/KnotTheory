#By the convention, going down is represented by a 1 in the code, but is still -q^{-1} in the final matrix.

import numpy as np
import re
import operator

"""
content

Input:
p - partition

Output:
the content of the partition
"""
def content(p):
    cont = 0
    for i in range(len(p)):
        cont += p[i] * i
        cont -= p[i] * (p[i]-1)/2
    return cont

"""
sgn

Input:
a - any integer

Output:
1 if a is positive, -1 if it is negative, 0 if it is 0.
"""

def sgn(a):
    if(a == 0):
        return 0
    return abs(a)/a

"""
findPaths

Input:
root - The current partition.
depth - Depth still to go in the tree.
path - The path taken to get to where we are.
proj - The partition for the representation.
hitproj - A boolean representing whether the representation has been reached in the path.
last - The last partition encountered in the path.

Output: A list of data for each path from the root node down the tree of length depth, consisting of a tuple of first the leaf, whether it contains the representation, and then the path down the tree.
"""

def findPaths(root, depth, path, proj, hitproj, last, statepath):
    pathList = []
    newroot = root[:]
    #Base case
    if(depth == 0):
        if newroot != proj:
            return [[root] + [hitproj] + path + [statepath]];
        else:
            return [[root] + [True] + path + [statepath]];

    downmoveplaces = list(set(newroot)) #The number of distinct values in the permutation is also equal to the number of places at which you can move down.

    rightmoveplaces = list(set(transpose(newroot))) #The number of distinct rows is also equal to the number of places at which you can move right.

    #oneDepthLower will be a list of tuples of first the next node down the tree, and then whether a move right or down was used to get there.
    oneDepthLower = []

    for i in downmoveplaces:
        #Add a down-move to the current path
        pathSoFar = [1] + path[:]
        temproot = root[:]

        #Execute the down-move and find the partition that results
        temproot[temproot.index(i)] += 1

        #Has the representation partition been encountered?
        hitproj2 = hitproj or (temproot == proj) or (root == proj) #The second case is for the fundamental representation
        if(path == []):
            pathList += (findPaths(temproot, depth - 1, pathSoFar, proj, hitproj2, root[:], [temproot] + statepath[:]))

        #Generate the list of ways we can move starting from the root, and which direction to move from the root to get there.
        oneDepthLower += [[temproot, 1]]


    for i in rightmoveplaces:
        #Each part here is the same except that to add to a column, we transpose the partition to turn columns into rows, add to the row, and transpose back to turn rows into columns.
        pathSoFar = [-1] + path[:]
        temproot = transpose(root[:])

        temproot[temproot.index(i)] += 1
        hitproj2 = hitproj or (transpose(temproot[:]) == proj) or (root == proj)

        if(path == []):
            pathList += (findPaths(transpose(temproot[:]), depth - 1, pathSoFar, proj, hitproj2, root[:], [transpose(temproot[:])] + statepath[:]))

        oneDepthLower += [[transpose(temproot[:]), -1]]

    if(path != []):
        test = oneDepthLower

    #Done with path-finding.

        #convert the list of unhashable lists to a list of tuples, which are hashable.
        for i in range(len(test)):
            test[i] = (tuple(test[i][0]), test[i][1])

        #Selecting the ambiguous cases. These are the ones that pass through the same 2 subsequent partitions but go there in two different ways.
        sub = list(set(filter(lambda a: len(list(filter(lambda b: sgn(b[1]) == -1 * sgn(a[1]) and a[0] == b[0], test))) != 0, test)))

        #Remove the ambiguous cases using contents.
        for i in sub:
            lookingFor = -1 * sgn(content(last) + content(i[0]) - content(root) - content(root))
            for j in range(len(oneDepthLower) - 1, -1, -1):
                #Eliminate ambiguous cases
                if(list(oneDepthLower[j][0]) == list(i[0]) and sgn(oneDepthLower[j][1]) != lookingFor):
                    del oneDepthLower[j]

        #Do recursion to find the rest of the paths
        for i in oneDepthLower:
            pathList += (findPaths(list(i[0]), depth - 1, [i[1]] + path[:], proj, (hitproj or (list(i[0]) == proj)), root[:], [list(i[0])] + statepath[:]))
    return pathList

"""
transpose

Input:
tuperm - array of the number of blocks in each row.

Output:
array corresponding to the number of blocks in each column.
"""
def transpose(tuperm):
    perm = list(tuperm[:])
    if(perm == []):
        return []
    else:
        returnValue = [0] * max(perm)
        currentIndex = 1
        currentValue = perm[0]

        #To transpose the partition, we add a block into the first val columns, where val is the number of blocks in a row, because there can be no gaps.
        for _, val in enumerate(perm):
           for i in range(val):
                returnValue[i] += 1
    try:
        return returnValue[0:returnValue.index(0)]
    except:
        return returnValue

def allBut(a, b):
    testa = a[:]
    return testa[:b] + testa[b + 1:]

n = int(input("How many strands?"))
cables = int(input("Cabled how many times?"))
rep = str(input("What (comma-separated) representation?")).replace("(", "").replace(")", "")
formattedRep = list(map(int, str(rep).replace(" ", "").split(",")))
n = n * cables - 1

#Find all paths at depth n
test = findPaths([1], n, [], formattedRep, False, [], [[1]])

#Transpose the matrix to retrieve the matrices we want
test2 = np.array(test[:]).T.tolist()

#Move the projection matrix to the end so it isn't used in sorting, as well as reversing the list so that the first move is used first to sort

test2 = [test2[0]] + [np.array(test2[-1]).T[::-1].T.tolist()] + [np.array(test2[2:-1][::-1]).T.tolist()] + [test2[1]]

#Sort the paths first by the partition at the end and then by the path to get there

test4 = np.array(test2).T.tolist()
test4.sort()
test2 = np.array(test4).T.tolist()
test4 = test2
test3 = test4[:]

#Find the number of times each partition appears; necessary to generate the D-Matrix
toLookFor = test4[0][0]
counter = 0
occurences = []

for i in range(len(test4[0])):
    if(test4[0][i] == toLookFor):
        counter += 1
    else:
        occurences += [counter]
        counter = 1
        toLookFor = test4[0][i]
occurences += [counter]

test2 = np.array(test2).T.tolist()

#Find the doublets and mark them.
for i in range(len(test2) - 1):
    for k in range(i + 1, len(test2) - 1):
        for j in range(1, n):
            if(type(test2[i][2][j]) != tuple and sgn(test2[i][2][j]) == 1 and sgn(test2[k][2][j]) == -1 and allBut(test2[i][1], j) == allBut(test2[k][1], j)):
                last = test2[i][1][j - 1]
                temproot = test2[i][1][j + 1]
                lastTransed = transpose(last[:])
                testlast = (last[:] + (len(temproot) - len(last)) * [0])
                testlastTransed = (lastTransed + (len(transpose(temproot[:]))-len(lastTransed)) * [0])
                if(2 not in map(operator.sub, temproot, testlast) and 2 not in map(operator.sub, transpose(temproot[:]), testlastTransed)):
                    isDoublet = True
                    ind1 = [a for a, x in enumerate(map(operator.sub, temproot, testlast)) if x == 1]
                    ind2 = [a for a, x in enumerate(map(operator.sub, transpose(temproot[:]), testlastTransed)) if x == 1]
                    hookLen = abs(ind1[1] - ind1[0]) + abs(ind2[1] - ind2[0])
                    test2[i][2][j] = (k - i, hookLen, i + 1)
                    test2[k][2][j] = 0

test2 = np.array(test2).T.tolist()

temp = test[2]
test[2] = test[3]
test[3] = temp

#Mathematica Code

var('q A')

def quant(c):
    return c - 1 / c

def p(n):
    return quant(A ^ n) / quant(q ^ n)

def pstar(n):
    return quant(A * q ^ (n - 1)) / quant(q ^ n)

def Schur(x, c):
    return schurmat(x, c).determinant()

def schurmat(x, c):
    m = matrix(len(x), len(x), lambda i, j: A)

    for i in range(len(x)):
        for j in range(len(x)):
            m[i, j] = h(x[j] - j + i, c)

    return m

def h(n, t):
    emm = Partitions(n)
    c = 0

    for j in emm:
        d = {}
        
        for i in j:
            if i not in d:
                d[i] = 0
            d[i] += 1
        
        tempor = 1
        
        for ell in d.keys():
            tempor2 = d[ell]
            tempor *= (p(t * ell) ^ (tempor2) / (ell ^ tempor2 * factorial(tempor2)))
        
        c = c + tempor

    return c
    
print(p(1))
print(Schur([2], 1).factor())

proj = diagonal_matrix(test3[3])

print(proj)

partitions = test3[0]

dDiagonal = []

for j in len(partitions):
    dDiagonal = dDiagonal + occurences[j] * [Schur(partitions[j],1)]

dMatrix = diagonal_matrix(dDiagonal)

#By the convention, going down is represented by a 1 in the code, but is still -q^{-1} in the final matrix.

import numpy as np
import re

"""
findPaths

Input:
root - The last visited node in the tree
depth - Depth still to go in the tree
path - The path taken to get to the root node
proj - The partition for the representation
hitproj - A boolean representing whether the partition has been reached in the path.

Output: A list of data for each path from the root node down the tree of length depth, consisting of a tuple of first the leaf, whether it contains the representation, and then the path down the tree.
"""

def findPaths(root, depth, path, proj, hitproj):
    newroot = root[:]

    #Base case
    if(depth == 0):
        if newroot != proj:
            return [[root] + [hitproj] + path];
        else:
            return [[root] + [True] + path];

    downmoveplaces = list(set(newroot)) #The number of distinct values in the permutation is also equal to the number of places at which you can move down.

    rightmoveplaces = list(set(transpose(newroot))) #The number of distinct rows is also equal to the number of places at which you can move right.

    pathList = []

    #oneDepthLower will be a list of tuples of first the next node down the tree, and then whether a move right or down was used to get there.
    oneDepthLower = []

    for i in downmoveplaces:
        #Add a down-move to the current path
        pathSoFar = [1] + path[:]
        temproot = root[:]

        #Execute the down-move and find the partition that results
        temproot[temproot.index(i)] += 1

        #Has the representation partition been encountered?
        hitproj2 = hitproj or (temproot == proj)

        if(path == []):
            pathList += (findPaths(temproot, depth - 1, pathSoFar, proj, hitproj2))

        #Generate the list of ways we can move starting from the root, and which direction to move from the root to get there.
        oneDepthLower += [[temproot, 1]]

    for i in rightmoveplaces:
        #Each part here is the same except that to add to a column, we transpose the partition to turn columns into rows, add to the row, and transpose back to turn rows into columns.
        pathSoFar = [-1] + path[:]
        temproot = transpose(root[:])
        temproot[temproot.index(i)] += 1
        hitproj2 = hitproj or (transpose(temproot[:]) == proj)

        if(path == []):
            pathList += (findPaths(transpose(temproot[:]), depth - 1, pathSoFar, proj, hitproj2))

        oneDepthLower += [[transpose(temproot[:]), -1]]

    if(path != []):
        test = oneDepthLower

        #convert the list of unhashable lists to a list of tuples, which are hashable.
        for i in range(len(test)):
            test[i] = (tuple(test[i][0]), test[i][1])

        #Selecting the ambiguous cases, because they are the ones where with two different last moves the same final position is reached.
        sub = list(set(filter(lambda a: test.count((a[0], -a[1])) != 0, test)))

        #Remove the ambiguous cases (i.e. when it is possible to go from one state to another in two different ways) by using the convention of q -> -q^{-1} and -q^{-1} -> q
        for i in sub:
            lookingFor = -1 * path[0]
            for j in range(len(oneDepthLower) - 1, -1, -1):
                #Eliminate ambiguous cases that are q -> q or -q^{-1} -> -q^{-1}
                if(list(oneDepthLower[j][0]) == list(i[0]) and oneDepthLower[j][1] != lookingFor):
                    del oneDepthLower[j]

        #Do recursion to find the rest of the paths
        for i in oneDepthLower:
            pathList += (findPaths(list(i[0]), depth -  1, [i[1]] + path[:], proj, (hitproj or (list(i[0]) == proj))))
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

n = input("How many strands?")
n = n - 1

print("\n Tree depth: " + str(n))

#Find all paths at depth n
test = findPaths([1], n, [], [2], False)

#Transpose the matrix to retrieve the matrices we want
test2 = np.array(test[:]).T.tolist()

#Move the projection matrix to the end so it isn't used in sorting, as well as reversing the list so that the first move is used first to sort
test2 = [test2[0]] + [np.array(test2[2:][::-1]).T.tolist()] + [test2[1]]

#Sort the paths first by the partition at the end and then by the path to get there
test2 = np.array(test2).T.tolist()
test2.sort()

test2 = np.array(test2).T.tolist()
test3 = test2[:]

#Find the number of times each partition appears; necessary to generate the D-Matrix
toLookFor = test2[0][0]
counter = 0
occurences = []
for i in range(len(test2[0])):
    if(test2[0][i] == toLookFor):
        counter += 1
    else:
        occurences += [counter]
        counter = 1
        toLookFor = test2[0][i]
occurences += [counter]

#Mathematica Code

print("""(*Prerequisites*)
        n[a_] := (q^a - q^(-a))/(q - q^(-1))
        k[g_] := (A (q^(g - 1)) - (A (q^(g - 1)))^(-1))/(q^g - q^(-g))
        h[n_] := If[n >= 0, Product[k[j], {j, 1, n}], 0]
        B[a_] := {{-1/((q^a) (n[a])), -Sqrt[n[a - 1] n[a + 1]]/
    n[a]}, {-Sqrt[n[a - 1] n[a + 1]]/n[a], q^a/n[a]}}
        DirSum[c_] := ArrayFlatten@ReleaseHold@DiagonalMatrix[Hold /@ c]
        DPartM[a_] :=
        Table[Table[h[Part[a, y] - y + x], {x, 1, Length[a]}], {y, 1,
   Length[a]}]
        DPart[a_] := Det[DPartM[a]]
    (*This ends the prerequisites cell, just containing basic definitions*)

    """)
print("(*The below is also boilerplate, but it is specific to the number of strands*)")
temp = list(set(map(lambda a: tuple(a), test3[0])))
temp.sort()
test3[0] = map(lambda a: list(a), temp)
print("Partitions = " + str(test3[0]).replace("[", "{").replace("]", "}"))
print("Paths = " + str(occurences).replace("[", "{").replace("]", "}"))
print("""(*End second boilerplate cell*)

    (*Actual computations begin here*)
    DMatrix = Simplify[DirSum[Join @@ Table[Table[DPart[Part[Partitions, a]], Part[Paths, a]], {a, 1, Length[Partitions]}]]]
      """)
projmatr = test3[2]
print("Proj = DirSum[" + str(map(int, projmatr)).replace("[", "{").replace("]", "}") + "]\n")
#Change the numbers into q and -q^{-1} and replace each instance of "-q^{-1}, q" with the corresponding B-Matrix except in R1.
test2 = np.array(test2[1: -1][0]).T.tolist()
for i in range(0, len(test2)):
    for j in range(len(test2[i])):
        test2[i][j] = "q" if test2[i][j] == -1 else "-q^(-1)"
    qList = str(test2[i]).replace("'","").replace("[","").replace("]","")
    finalOutput = qList if i == 0 else re.sub("-q\^\(-1\), q", "B[" + str(i + 1) + "]", qList)
    print("R" + str(i + 1) + " = DirSum[{" + str(finalOutput) + "}]")
    print("")

print("""
(*End computation cell*)

(*Begin testing cell*)
""")
for i in range(0, len(test2)):
    print("Simplify[Tr[DMatrix.R" + str(i + 1) + "] - (-(((-1 + A^2)^"+ str(n) +" q^"+ str(n) +")/(A^"+ str(n + 1) +" (-1 + q^2)^"+ str(n) +")))]")
    print("Simplify[Tr[DMatrix.R" + str(i + 1) + "] / (-(((-1 + A^2)^"+ str(n) +" q^"+ str(n) +")/(A^"+ str(n + 1) +" (-1 + q^2)^"+ str(n) +")))]")
print("""
(*Begin testing cell*)
""")
for i in range(0, len(test2)):
    print("Simplify[R" + str(i + 1) + "]")
print("""
(*Begin testing cell*)
""")
for i in range(0, len(test2) - 1):
    print("MatrixForm[Simplify[R" + str(i + 1) + ".R" + str(i + 2) + ".R" + str(i + 1) + "-" + "R" + str(i + 2) + ".R" + str(i + 1) + ".R" + str(i + 2) + "]]")

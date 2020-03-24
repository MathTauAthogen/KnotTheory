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
#Mathematica Code

print("(*Prerequisites*)")
print("pathNum = " + str(sum(occurences)))

print("""
        rep={""" + str(formattedRep).replace("[", "").replace("]","") + """}
        bra[x_]:=x-x^(-1)
        q[l_, i_] := (m = Table[Table[0, pathNum], pathNum]; If[i == 1, m[[l, l]] = q, m[[l,l]] = -q^(-1)]; Return[m])
        n[a_] := (q^a - q^(-a))/(q - q^(-1))
        k[g_] := (A (q^(g - 1)) - (A (q^(g - 1)))^(-1))/(q^g - q^(-g))
        h[n_] := If[n >= 0, Product[k[j], {j, 1, n}], 0]
        B[g_, a_, b_] := (m = Table[Table[0, pathNum], pathNum];
        m[[b, b]] = -q^(-a)/n[a];
        m[[b, b + g]] = -Sqrt[n[a - 1] n[a + 1]]/n[a];
        m[[b + g, b]] = -Sqrt[n[a - 1] n[a + 1]]/n[a];
        m[[b + g, b + g]] = q^(a)/n[a];
        Return[m])
        DirSum[c_] := ArrayFlatten@ReleaseHold@DiagonalMatrix[Hold /@ c]
        SumMatrs[c_] := (m = Table[Table[0, pathNum], pathNum]; Do[m = m + elem, {elem, c}]; Return[m])
        DPartM[a_, t_] :=
        Table[Table[h[t(Part[a, y] - y + x)], {x, 1, Length[a]}], {y, 1,
   Length[a]}]
        SchurPoly[a_] := Det[DPartM[a, 1]]
        SchurPolyMult[x_, c_] := Det[DPartM[x, c]]
    (*This ends the prerequisites cell, just containing basic definitions*)
    """)
print("(*The below is also boilerplate, but it is specific to the number of strands*)")
temp = list(set(map(lambda a: tuple(a), test3[0])))
temp.sort()
test3[0] = map(lambda a: list(a), temp)

print("Partitions = " + str(list(test3[0])).replace("[", "{").replace("]", "}"))
print("Paths = " + str(occurences).replace("[", "{").replace("]", "}"))
print("""(*End second boilerplate cell*)

    (*Actual computations begin here*)
    DMatrix = Simplify[DirSum[Join @@ Table[Table[SchurPoly[Part[Partitions, a]], Part[Paths, a]], {a, 1, Length[Partitions]}]]]
      """)
projmatr = test3[3]

print("Proj = DirSum[" + str(list(map(int, projmatr))).replace("[", "{").replace("]", "}") + "]\n")
print("ComputePoly[c_] := (m = Proj.DMatrix; Do[m = m.elem, {elem, c}]; Return[Tr[m]])")

#Change the numbers into q and -q^{-1} and replace each instance of "-q^{-1}, q" with the corresponding B-Matrix except in R1.
test2 = np.array(test2[2: -1][0]).T.tolist()
for i in range(0, len(test2)):
    for j in range(len(test2[i])):
        test2[i][j] = "q[" + str(j + 1) + ", 1]" if test2[i][j] == -1 else ( "q[" + str(j + 1) + ", -1]" if test2[i][j] == 1 else test2[i][j])
    qList = str(test2[i]).replace("'","").replace("[q","q").replace("]]","]").replace("[B","B")
    finalOutput = re.sub(", 0", "", qList)
    finalOutput = re.sub("[^\^]\((.*?)\)", "B[\\1]", finalOutput)
    print("R" + str(i + 1) + " = SumMatrs[{" + str(finalOutput) + "}]")
    print("")
matrixStr = "r = Simplify[{"
for i in range(int((n + 1)/cables) - 1):
    if(cables != 1):
        cabled = []
        center = cables * (i + 1)
        for j in range(1, cables):
            for k in range(center - j, center + j + 1, 2):
                cabled += [k]
        cabled = cabled + cabled[::-1][int((n+1)/cables):] + [center]
        cabledStr = "R"+str(center)
        for j in cabled:
            cabledStr += ".R"+str(j)
        matrixStr += (cabledStr + ", ")
    else:
        matrixStr += ("R"+str(i + 1) + ", ")
matrixStr = matrixStr[:-2]
matrixStr += "}]"
print(matrixStr)
matrixStr = "rinv = Simplify[{"
for i in range(int((n + 1)/cables) - 1):
    if(cables != 1):
        cabled = []
        center = cables * (i + 1)
        for j in range(1, cables):
            for k in range(center - j, center + j + 1, 2):
                cabled += [k]
        cabled = cabled + cabled[::-1][int((n+1)/cables):] + [center]
        cabledStr = "Inverse[R"+str(center)+"]"
        for j in cabled:
            cabledStr += ".Inverse[R"+str(j)+"]"
        matrixStr += (cabledStr + ", ")
    else:
        matrixStr += ("Inverse[R"+str(i + 1) + "], ")
matrixStr = matrixStr[:-2]
matrixStr += "}]"
print(matrixStr)
print("""
PolyFromBraidWord[c_] := (matrixList = {}; Do[matrixList = Append[matrixList, If[elem > 0,r[[elem]],rinv[[-elem]]]], {elem, c}];Return[Simplify[ComputePoly[matrixList]]])
CommonDenom = SchurPoly[rep]
NormalizePoly[x_] := Factor[PolyFromBraidWord[x]/CommonDenom]
CoefSimplify[c_] := Factor[CoefficientList[NormalizePoly[c], A]]
Content[c_] := (cont = 0;Do[cont = cont + c[[i]] * i;cont = cont - c[[i]]*c[[i-1]]/2,{i, Length[c]}]; Return[cont])
Torus[m_,n_] := (poly = 0;Do[q^(-2*n*Content[c]*SchurPoly[c]/m)*SchurMult[c,m]]; Return[Poly])
(*End computation cell*)
""")

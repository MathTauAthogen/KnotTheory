#By the convention, going down is 1, but is still -q^{-1} in the final matrix.
#TODO: Insert proper code comments and conform to the Google standard
import numpy as np
import re

def findPaths(root, depth, path, proj, hitproj):
    newroot = root[:]
    if(depth == 0):
        if newroot != proj:
            return [[root] + [hitproj] + path];
        else:
            return [[root] + [True] + path];
    downmoveplaces = list(set(newroot)) #The number of distinct values in the permutation is also equal to the number of places that you can push down.
    rightmoveplaces = list(set(transpose(newroot)))
    pathList = []
    shallowPathList = []
    for i in downmoveplaces:
        pathSoFar = [1] + path[:]
        temproot = root[:]
        temproot[temproot.index(i)] += 1
        hitproj2 = hitproj or (temproot == proj)
        if(path == []):
            pathList += (findPaths(temproot, depth - 1, pathSoFar, proj, hitproj2))
        shallowPathList += [[temproot, 1]]
    for i in rightmoveplaces:
        pathSoFar = [-1] + path[:]
        temproot = transpose(root[:])
        temproot[temproot.index(i)] += 1
        hitproj2 = hitproj or (transpose(temproot[:]) == proj)
        if(path == []):
            pathList += (findPaths(transpose(temproot[:]), depth - 1, pathSoFar, proj, hitproj2))
        shallowPathList += [[transpose(temproot[:]), -1]]
    if(path != []):
        test = shallowPathList
        for i in range(len(test)):
            test[i] = (tuple(test[i][0]), test[i][1])
        sub = list(set(filter(lambda a: test.count((a[0], 1)) + test.count((a[0], -1)) > 1 and test.count((a[0], 1)) != test.count((a[0], -1)) + test.count((a[0], -1)), test)))
        for i in sub:
            lookingFor = -1 * path[0]
            for j in range(len(shallowPathList) - 1, -1, -1):
                if(list(shallowPathList[j][0]) == list(i[0]) and shallowPathList[j][1] != lookingFor):
                    del shallowPathList[j]
        for i in shallowPathList:
            pathList += (findPaths(list(i[0]), depth -  1, [i[1]] + path[:], proj, (hitproj or (list(i[0]) == proj))))
    pathList.sort()
    return pathList

def transpose(tuperm):
    perm = list(tuperm[:])
    if(perm == []):
        return []
    for i in range(len(perm)):
        perm[i] = perm[i] - 1
    return [len(perm)] + transpose(filter(lambda a: a !=  0, perm))

n = 4
print("\n Tree depth: " + str(n))
partitionOrder= [[5],[4,1],[3,2],[3,1,1],[2,2,1],[2,1,1,1],[1,1,1,1,1]][::-1]#[[6], [5, 1], [4, 2], [4, 1, 1], [3, 3], [3, 2, 1], [3, 1, 1, 1], [2, 2, 2], [2, 2, 1, 1], [2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]][::-1]
test = findPaths([1], n, [], [2], False)
test2 = np.array(test[:]).T.tolist()
projmatr = test2[1]
test2 = [test2[0]] + [np.array(test2[2:][::-1]).T.tolist()] + [test2[1]]
test2 = np.array(test2).T.tolist()
test2.sort()
test2 = [a for b in partitionOrder for a in test2 if a[0]==b]
test2 = np.array(test2).T.tolist()
print("The partitions: " + str(test2[0]))
print("")
toLookFor = test2[0][0]
counter = 0
for i in range(len(test2[0])):
    if(test2[0][i] == toLookFor):
        counter += 1
    else:
        print("The partition " + str(toLookFor) + " appears " + str(counter) + " times.")
        counter = 1
        toLookFor = test2[0][i]
print("The partition " + str(toLookFor) + " appears " + str(counter) + " times.")
print("")
print("Check: number of partitions = " + str(len(test2[0])))
print("")
test2 = np.array(test2[1:-1][0]).T.tolist()
for i in range(0,len(test2)):
    for j in range(len(test2[i])):
        test2[i][j] = "q" if test2[i][j] == -1 else "-q^(-1)"
    qList = str(test2[i]).replace("'","").replace("[","").replace("]","")
    finalOutput = qList if i == 0 else re.sub("-q\^\(-1\), q", "B[" + str(i + 1) + "]", qList)
    print("R" + str(i + 1) + ": " + str(finalOutput))
    print("")
print("Proj: " + str(map(int, projmatr)))

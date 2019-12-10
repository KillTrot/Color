from rubik_solver import utils
import rubicColorDetection as colorDe
import numpy as np

def getCubeColors():
    return colorDe.getColorFront()

def getCorrectCube():
    cubelets = []
    lastCubi = []
    i = 0
    while (i < 5):
        while len(cubelets) != 9:
            cubelets = getCubeColors()
        if i != 0:
            if lastCubi != []:
                if cubelets == lastCubi:
                    i+=1
                else:
                    i = 0
                    lastCubi = cubelets
            else:
                i = 0
                lastCubi = cubelets
        else:
            i = 1
            lastCubi = cubelets
    return cubelets

def getColorString():
    while len(cubelets) != 0:
        tempObj = []
        tempValue = 0
        for cube in cubelets:
            if cube[1] > tempValue:
                tempValue = cube[1]
                tempObj = cube
        if len(cubelets) == 3:   
            cube_row = cubelets
        else:
            cube_row.append(tempObj)
            cubelets.remove(tempObj)
        if len(cube_row) == 3:
            while len(cube_row) != 0:
                tempObj = []
                tempValue2 = 0
                for cube2 in cube_row:
                    if cube2[0] > tempValue2:
                        tempValue2 = cube2[0]
                        tempObj2 = cube2
                if tempValue2 == 0:
                    if len(charArray) == 0:
                        charArray.append(tempObj2[2])
                        cube_row = []
                    else:
                        i = len(charArray)
                        charArray.insert(i,tempObj2[2])
                else:
                    if len(charArray) == 0:
                        charArray.append(tempObj2[2])
                    else:
                        i = len(charArray)
                        charArray.insert(i,tempObj2[2])
                    cube_row.remove(tempObj2)
    
            print(''.join(charArray))
            cube_row = []



#walkthrough = utils.solve(cube, 'Beginner')
#print(walkthrough)


colorDe.getTestColorFront()

cubeString = ""
charArray = []
cube_row = []

oneCube = getCorrectCube()
getColorString(oneCube)
##turn left
##trun down
##turn right
##turn right
##turn down



module quantumNumbers

export quantumNumbers2D
export getQuantumNumbers

const quantumNumbers1D = [0
1
2
3
4
5
6
7
8
9]

const quantumNumbers2D = [0 0
1 0 
0 1 
2 0 
1 1 
0 2 
3 0 
2 1 
1 2 
0 3]

const quantumNumbers3D = [0 0 0
1 0 0
0 1 0
0 0 1
2 0 0
0 2 0
0 0 2
1 1 0
1 0 1
0 1 1]

function getQuantumNumbers(col::Int64, numDimensions::Int64)
    if numDimensions == 1
        return quantumNumbers1D[col,:]
    elseif numDimensions == 2
        return quantumNumbers2D[col,:]
    elseif numDimensions == 3
        return quantumNumbers3D[col,:]
    else 
        println("Use dim = 2 or dim = 3")
    end
end 

end
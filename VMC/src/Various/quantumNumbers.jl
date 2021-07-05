module quantumNumbers

export quantumNumbers2D
export getQuantumNumbers

quantumNumbers2D = [0 0
                    1 0 
                    0 1 
                    2 0 
                    1 1 
                    0 2 
                    3 0 
                    2 1 
                    1 2 
                    0 3]

quantumNumbers3D = [0 0 0
                    1 0 0
                    0 1 0
                    0 0 1
                    2 0 0
                    0 2 0
                    0 0 2
                    1 1 0
                    1 0 1
                    0 1 1]

function getQuantumNumbers(col, numDimensions)
    if numDimensions == 2
        return quantumNumbers2D[col,:]
    else
        return quantumNumbers3D[col,:]
    end
end 

end
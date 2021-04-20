module jastrow 

export jastrowComputeRatio

function jastrowComputeRatio(system, old_position, particleMoved)
    changeInDistanceSum = 0
    new_position = system.particles
    for i=1:system.n_particles
        if i != particleMoved
            new_difference = system.particles[i, :] - system.particles[particleMoved, :]
            new_distance = sqrt(dot(new_difference, new_difference))

            old_difference = old_position[i, :] - old_position[particleMoved, :]
            old_distance = sqrt(dot(old_difference, old_difference))

            changeInDistanceSum += system.beta*(new_distance - old_distance)
        end 
    end

    return exp(2*changeInDistanceSum)
end 

end
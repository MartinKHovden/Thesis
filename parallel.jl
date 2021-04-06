using Distributed

addprocs(2)

function main()
    @everywhere println(myid())
end 

main()

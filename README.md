# InverseProblems.jl
Julia Repository for DL-Driven solutions to Inverse Problems

## Logistics

To address the elephant in the room, this is not a Julia package.
If anything, this repository houses a set of scripts that I try to automate as much as possible, to enable reproducibility.
That said, the work here shall be converted to reusable library standard code when time permits. But this is not my priority.

If you are new to Julia, there are somethings that you have to do first.

    - Install Julia
    - Install a VS Code extension to Julia (If you use another editor, explore and install the corresponding Julia extension.)

Then you are in good shape to start reproducing this code.

    - Instantiate the current development environment in Julia.
        - To do so: 
            - Press "]" on the Julia REPL
            - Type activate . (while you are in the cloned repository.)
        - This should fetch all the dependencies for the work here.
        
## About

This repository addresses the problem Waveform inversion of Ultrasonic Non-destructive testing signals. While this is a problem that is as old as time, our approach deviates from the previous adjoint based solvers, in that, a noise immune regularization network is used to faciliate the inversion of noisy data. Furthermore, the optimization step makes use of Surroagate models developed using the newly proposed Operator Regression Paradigm of Scientific Machine Learning, in contrast to the previous HPC based solutions.

So, the problem can be broken down into two parts : 

    - Developing a surrogate for ultrasonic wave propagation in a defect embedded domain
    - Optimizing for the defect using the Regularization network.
    
To this point, the code in the repository contains the solution to the first half of the problem. More details to follow ...

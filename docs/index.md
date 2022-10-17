@def title = "Franklin Sandbox"
@def hasmath = true
@def hascode = true


# Inverse Scattering Problem using Neural Operators

Inverse Scattering Problems are extremely important in Experimental Physics and several Engineering Applications. The premise of the problem involves discovering the properties of systems which are less accessible to direct experimentation. Typical examples include Seismography, Non-destructive testing and different forms of Tomography in medicine.

While analytical solutions to the scattering problem are available for a number of simple geometries, one has to resort to numerical approximation in real-world applications.

Numerically, the shape of the defect is reconstructed using optimization techniques. As a rule, this should involve several calls to the routine that solves the acoustic-wave equation in the domain of interest. This is a computationally expensive and tedious affair often requiring several minutes to solve one instance of the forward problem. Significant improvements could be realized with the help of suitable surrogate models. 

Neural Operators are a recent phenomena, that have seen rapid application and adoption among the Model Reduction commmunity. In this work we incorporate a Fourier Neural Operator to the optimization loop of the numerical inverse-scattering algorithm. Accuracy and Efficiency of the modified algorithm is reported. An account on the stability of the solution is also provided.

# Table of Contents

- [Literature](/literature/)
- [Data Generation](/data-gen/)
- [Fourier Neural Operator](/fno/)
- [Adversarial Regularization](/reg-net/)
- [Optimization](/optimization/)
- [Results](/results/)
- [Code](/code/)

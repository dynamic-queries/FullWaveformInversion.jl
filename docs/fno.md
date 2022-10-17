# Fourier Neural Operators
As detailed in the Simulation and DataGen section, there are two stages to the entire simulation.
- One where the sensors behave as sources. ~ Initial Surrogate
- The other where the sensors are receivers. ~ Time series Surrogate

## Initial surrogate

## Time series surrogates
Fourier Neural Operators, unlike DeepONets, do not allow us to provide parameters for time steps in the simulation.
As a result, one has to alternatively train 250 separate Neural Nets to predict the first 250 time steps which are then used in the optimization loop of the scattering problem.

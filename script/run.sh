#!/bin/bash

echo "-----------------------------------------------------------------------\n"
echo "                        Full Waveform Inversion                        \n"
echo "-----------------------------------------------------------------------\n\n\n"


echo "Generating data...\n"
echo "This solves 750 instances of the acoustic wave equation\n"
echo "Make sure you are running on a machine with atleast 25 cores.\n" 
mpiexecjl -n 25 julia script/datagen.jl

echo "The scattered data is now combined into a single bundle so that we can train the surrogate \n"
julia script/extract_data.jl

echo "We train the surrogates, one after the other...\n"
julia script/training/initial_surrogate/jl
julia script/training/os_surrogate.jl

echo "The best performing models are moved to a separate folder.\n"
julia script/best.jl

echo "The Full Waveform Inversion package is now ready to use. \n"

echo "\n\n-----------------------------------------------------------------------\n"
echo "                         END                                            \n"
echo "-----------------------------------------------------------------------"
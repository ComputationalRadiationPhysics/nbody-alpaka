# N-body problem with Alpaka
## Description
Our task is to simulate the [N-body problem](https://en.wikipedia.org/wiki/N-body_problem) with [Alpaka](https://github.com/ComputationalRadiationPhysics/alpaka).
## Playground
In the folder tests/simulationTest are two files for random generation of bodies and a visualizer. For the visualization you will need VPython for python 2

How to compile and use this:
```
# Go to folder
cd tests/simulationTest

# Generate makefile
cmake .

# Compile generator
make

# Generate data
./simulationClass_test2.out > test.txt

# Visualize data
python2 vision.py test.txt
```

## Schedule (in german)
We have our project schedule in markdown here: [Zeitplan](zeitplan.md)
## The team
We are two students from the TU-Dresden and chose this project in the context of the module "Hochparallele Simulationsrechnungen mit CUDA und OpenCL" (eng. highly parallel calculations for simulations with CUDA and OpenCL).
- Vincent Ridder, Informationssystemtechnik (mixed studies of EE and CS)
- Valentin Gehrke, Informationssystemtechnik (mixed studies of EE and CS)


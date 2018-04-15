# emugen

## What?

This is an open source version of emulator. Emulator is conceptually a simulator of the simulator but it can have a wider usage. Given any set of inputs/output combination, emulator can learn the non-linear relationship between the inputs and outputs. After the emulator is created, it can be queried with new unknown input parameter combinations and it will return a predicted output and standard deviation, incorporating distributional uncertainty about the underlying Gaussian process model assuming hyperparameters are fixed at the mean of their posterior distribution. Other outputs are also available, including the expectation and variance of basis weights used in the model formulation, samples from the output space, and point estimates of the overall variance of an output.

## Why?

Generally, simulating a simulator is a difficult problem due the existence of the non-linear relationships among the inputs and outputs. Since, simulators need to solve complex physics equations to generate an output, it is time consuming and supercomputers are needed. A fast and reliable emulator on the other hand can quickly generate a similar solution in much less time, there by helping the domain scientists in their research.

## How?

This software uses a Bayesian approach via rejection sampling to converge on the highly likely regions of high dimensional input space from the training samples. It uses Gaussian process-based Markov Chain Monte-Carlo (MCMC) for this purpose.

# Installation Guide

The ``egg`` Python package is installed by doing:
```
python setup.py install
```

There are two versions of the Gaussian Process MCMC. There is a serial version and a parallel version. Each version has different dependencies.

## Serial

Dependencies:
  * GSL
  * GSL CBLAS 

Please install these two libraries. You may need to change the ``Makefile`` in ``egg/gpmcmc/serial`` and ``egg/emu`` if the previously mentioned library paths are not in the default search path.

## Parallel

dependencies:
  * CUDA (a Nvidia GPU is required)
  * GSL
  * GSL CBLAS

Please install these libraries. You may need to change the ``Makefile`` in ``egg/gpmmcmc/gpu`` and ``egg/emu`` if the previously mentioned library paths are not in the default search path.

# Quickstart Usage Guide

A generic (serial or GPU) usage example is:
```
emu_create --design-file <designFile> --sim-file <simOutFile> --samples <mcmcSamples> [--options]
```

Here, ``designFile`` refers to a CSV file with a header specifying the input combinations.
This file has ``M`` rows that correspond to the different simulation runs, and ``N`` columns that correspond to the ``N`` inputs of the system.

``simOutFile`` refers to a CSV file with a header specifying the outputs for each input combination.
This file has ``M`` rows that correspond to the different simulation runs, and the number of columns ``K`` can be 1 if the simulation generates a single value or greater than 1 if the output is multi-valued.
A more detail explaination of the input file and output file format is provided in the section below.

``mcmcSamples`` refers to number of MCMC samples. Typically, this value should be 10,000 but less samples can be used for faster and less accurate emulator creation.

There are optional command line options to use GPUs. Appending ``--gpu`` to the end of the command specifies if Nvidia GPU will be used.

For example, a typical serial execution example might look like this:
```
emu_create --design-file examples/velocimetry/design.txt --sim-file examples/velocimetry/output.txt --samples 10000
```

If you are using the GPU implementation, then you should set the environment variable ``CUDA_PATH`` to your CUDA installation.
A typical GPU parallel execution example might look like this:
```
export CUDA_PATH=/projects/opt/centos7/cuda/8.0
emu_create --design-file examples/velocimetry/design.txt --sim-file examples/velocimetry/output.txt --samples 10000 --gpu
```

# The ``egg`` Python Package

## Wrapper functions for constructing an emulator

The purpose of the ``egg`` Python package is to provide convenient wrappers around the MCMC and emulation codes such that its easier for the user to construct their own pipelines with the emulator.
An example is the executable located at ``src/python/emu_create``.
You mainly need these two functions as demonstrated in ``emu_create``: 
  * ``egg.hatch.hatch`` : This function creates the emulator.
  * ``egg.fly.fly`` : This function predicts based on the already created emulator.

If you have the same data, you only need to call the ``egg.hatch.hatch`` function once which is more time consuming part of the emulator.
Then, you can repeatedly call the ``egg.fly.fly`` function which predicts the simulation output for the new input sequences.

It is important to note that the MCMC and emulation codes themselves are standalone C/C++ executables which are compiled and called by ``egg`` modules such as ``egg.hatch`` and ``egg.fly`` in a temporary directory.
If you want to run multiple instances of these calls, then you should run each call in a separate directory because ``emu_create`` will write out intermeditary file to disk as part of the pipeline.
You should also note that re-running ``emu_create`` in the same directory will overwrite the existing files.

## ``fly`` input options and requirements

If you have the same data and have already called the ``egg.hatch.hatch`` function (either directly or using the above example), then you can call the ``egg.fly.fly`` function for new input sequences.
This function has one required argument ``input_file`` and two optional arguments ``n_samples`` and ``burn_in``: 
  * ``input_file`` : This argument specifies the name of the file containing new input sequences. This should be a ASCII ``.txt`` file with each new input on its own line separated by spaces.
  * ``n_samples`` : This argument defines how many random samples the user would like to generate per each input line. Note that ``n_samples`` is required to be a positive integer. The default value for ``n_samples`` is 1 if the user does not input a value or if the user inputs a value that is not a positive integer.
  * ``burn_in`` : This argument defines what fraction of the MCMC chain one wishes to discard (first ``burn_in * 100`` percent of samples are discarded). The user must input a value less than 1 and greater than or equal to 0. The default value for burn in (if the user does not input a value or enters a value that is not allowed) is 0.25, leading to the first 25% of MCMC draws to be discarded. The user should assess the convergence of the chain to determine the appropriate value for burn in.

## Input file format

The ``egg`` Python package uses a Cinema database as its input file format.
A Cinema database is a comma-separated value (CSV) file with a header for each column.
For example, a Cinema database might look like:
```
x0,x1,x2,x3
1.0,2.0,3.0,4.0
1.0,4.0,9.0,16.0
1.0,16.0,81.0,256.0
...
```

An example dataset formatted as a Cinema database is provided in ``examples/velocimetry``.
There are two files, an input file that holds the input parameters given to the simulation and an output file that holds the output of the simulation for those given input parameters.
The input file called ``design.txt`` has a column for each input parameter to the simulation, and each row corresponds to a particular set of input parameters used to generate a simulation.
For a given row in the input file, the output of the simulation has the same row index in the output file called ``output.txt``.
For example, the parameters from the ``i``-th row in the input file were used to generate the simulation output in the ``i``-th of the output file.


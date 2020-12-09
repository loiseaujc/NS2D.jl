# NS2D

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/loiseaujc/NS2D.jl.svg?branch=master)](https://travis-ci.com/loiseaujc/NS2D.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/loiseaujc/NS2D.jl?svg=true)](https://ci.appveyor.com/project/loiseaujc/NS2D-jl)
[![Coverage](https://codecov.io/gh/loiseaujc/NS2D.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/loiseaujc/NS2D.jl)

![Julia flow](imgs/anim_vorticity.gif)

---

## Overview

NS2D is a simple pseudo-spectral solver for the two-dimensional Navier-Stokes equations in a doubly periodic domain written in Julia.
Its aim is primarily to get myself familiar with the Julia ecosystem for scientific computing as well as illustrating how pseudo-spectral solvers work for my students.
If you nonetheless think that such a simple solver written in Julia could be beneficial for your research, please do not hesitate to use it, fork it and/or send pull requests !

## Method / Capabilities

NS2D solves the two-dimensional Navier-Stokes equations in a doubly-periodic domain using the vorticity-streamfunction formulation.
Most of the computation is performed in spectral space while the non-linear advection term is computed in physical space using standard techniques for dealiasing.
At its core, NS2D uses `FFTW.jl` with multi-threading and preallocated arrays for in-place fast Fourier transforms while time-integration is performed thanks to `DifferentialEquations.jl`.

## Examples

### Taylor-Green vortices

TBA

### Two-dimensional decaying isotropic turbulence

TBA

### Two-dimensional Kolmogorov flow

TBA

## Developments goals

As stated, NS2D is primary a sandbox for me to get familiar with the Julia ecosystem and test some ideas.
Below is a non-exhaustive lists of such ideas.
If you are interested in any of them, please do not hesitate to get in touch !

- Deterministic and stochastic forcing for forced turbulence,
- Parallel computing,
- GPU acceleration with `CUDA.jl`,
- Automatic differentiation with `JuliaDiff` for optimization and machine-learning,

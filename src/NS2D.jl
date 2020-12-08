module NS2D

    # --> Fast Fourier Transform.
    using FFTW
    FFTW.set_num_threads(Threads.nthreads())

    # --> ODE solvers.
    using OrdinaryDiffEq

    # --> Misc.
    using Parameters
    using Statistics

    # --> Misc.
    include("utils.jl")
    export compute_reynolds_number, compute_velocity, compute_taylor_scale

    # --> Initial conditions related.
    include("ic.jl")
    export isotropic_turbulence

    # --> Computational core.
    include("core.jl")
    export SimParams, rhs!, simulate

end

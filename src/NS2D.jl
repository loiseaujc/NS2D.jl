module NS2D

    # --> Fast Fourier Transform.
    using FFTW
    FFTW.set_num_threads(Threads.nthreads())

    # --> ODE solvers.
    using OrdinaryDiffEq

    # --> Misc.
    using Parameters
    using Statistics

    # --> Computational core.
    include("core.jl")
    export SimParams, UnforcedProblem, simulate

    # --> Kolmogorov Flow.
    include("kolmogorov.jl")
    export KolmogorovFlow

    # --> Initial conditions related.
    include("ic.jl")
    export isotropic_turbulence, isolated_vortex, dipole, taylor_green

    # --> Misc.
    include("utils.jl")
    export compute_reynolds_number, compute_velocity, compute_taylor_scale, mesh

end

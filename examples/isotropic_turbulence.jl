using NS2D
using Plots
using FFTW
using DifferentialEquations

function plot_vorticity(ω, p)

    u, v = compute_velocity(ω, p)

    ω = real(ifft(ω))

    x = LinRange(-p.L/2, p.L/2, p.n+1)[1:end-1]

    fig = heatmap(
        x, x, ω,
        aspect_ratio=:equal,
        c=cgrad(:RdBu),
        xlims=(-p.L/2, p.L/2),
        ylims=(-p.L/2, p.L/2),
        clims=(-maximum(abs.(ω)), maximum(abs.(ω)))
    )

    return fig
end

function plot_velocity(ω, p)

    #
    u, v = compute_velocity(ω, p)

    #
    velocity_magnitude = .√(u.^2 + v.^2)

    x = LinRange(-p.L/2, p.L/2, p.n+1)[1:end-1]

    fig = heatmap(
        x, x, velocity_magnitude,
        aspect_ratio=:equal,
        c=cgrad(:viridis),
        xlims=(-p.L/2, p.L/2),
        ylims=(-p.L/2, p.L/2),
        clims=(0, maximum(velocity_magnitude)),
    )

    return fig
end

function main(; T=1)

    # --> Simulation parameters (default).
    p = SimParams()

    # --> Isotropic enstrophy spectrum
    #     (unit energy for all Fourier modes)
    spectrum(x) = one(x)
    ω = isotropic_turbulence(spectrum, p)

    # -->
    prob = UnforcedProblem(p, ω)

    # --> Callback.
    data = SavedValues(Float64, Float64)
    cb = SavingCallback((u, t, integrator) -> maximum(abs.(u)), data)

    # --> Keyword arguments for DifferentialEquations.jl
    kwargs = Dict(
        :saveat => 0:0.1:T,
        :callback = > cb
    )

    # --> Run the simulation.
    sol = simulate(prob, T ; kwargs...)

    # --> Plot the final condition.
    ω = sol[end]
    fig1 = plot_vorticity(ω, p)
    title!("Vorticity")

    fig2 = plot_velocity(ω, p)
    title!("Kinetic energy")

    fig = plot(fig1, fig2, layout=(1, 2), size=(800, 400))
    display(fig)

    return sol, data
end

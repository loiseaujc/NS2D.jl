using NS2D
using Plots
using FFTW
using DifferentialEquations
using Statistics

function main(; n=256, ν=1e-3, T=10)

    # --> Parameters for the simulation.
    p = SimParams(n=n, ν=ν)

    # --> Taylor-Green initial condition.
    ω₀ = convert(Array{Complex{Float32}, 2}, taylor_green(p))

    # -->
    prob = UnforcedProblem(p, ω₀)

    # --> Time-dependent analytic solution.
    ground_truth(t) = ω₀ .* exp(-2ν*t)

    # --> Save the error between the simulation and the analytic solution
    #     over time in the L_inf norm.
    data = SavedValues(Float64, Float64)
    cb = SavingCallback(
        (ω, t, integrator) -> maximum(abs.(ifft(ω - ground_truth(t)))),
        data
    )

    # --> Keyword arguments for DifferentialEquations.jl
    kwargs = Dict(
        :save_everystep => false,
        :callback => cb,
        :abstol => 1e-8,
        :reltol => 1e-6,
    )

    # --> Run the simulation.
    simulate(prob, T ; kwargs...)

    # --> Plot the Taylor-Green vorticity distribution.
    x, y = mesh(p)

    p₁ = heatmap(
        x, y, real(ifft(ω₀)),
        c=:RdBu,
        clims=(-2, 2),
        xlims=(-p.L/2, p.L/2),
        ylims=(-p.L/2, p.L/2),
        aspect_ratio=:equal,
    )

    display(p₁)
    savefig("taylor_green_vortices.png")

    # --> Plot the L_inf norm.
    p₂ = plot(
        data.t, data.saveval,
        xlims=(0, T),
        ylims=(0, 1e-9),
        legend=false,
        xlabel="Time",
        ylabel="Error",
    )

    display(p₂)
    savefig("taylor_green_vortices_error.png")

    return
end

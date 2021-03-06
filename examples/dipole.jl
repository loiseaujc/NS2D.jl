using NS2D
using Plots
using FFTW

function plot_vorticity(ω, p)

    u, v = compute_velocity(ω, p)

    ω = real(ifft(ω))

    x, y = mesh(p)

    fig = heatmap(
        x, y, ω,
        aspect_ratio=:equal,
        c=cgrad(:RdBu),
        xlims=(-p.Lx/2, p.Lx/2),
        ylims=(-p.Ly/2, p.Ly/2),
        clims=(-maximum(abs.(ω)), maximum(abs.(ω))),
        legend=false,
        size=(400, 400),
        axis=nothing,
        showaxis=false,
    )

    return fig
end

function main(; Lx=2π, Ly=2π, nx=256, ny=256, ν=1e-3, T=10)

    # --> Parameters for the simulation.
    p = SimParams(Lx=Lx, Ly=Ly, nx=nx, ny=ny, ν=ν)

    # --> Dipole initial condition.
    ω = dipole(p)

    # --> Keyword arguments for DifferentialEquations.jl
    kwargs = Dict(
        :saveat => 0:0.5:T
    )

    # --> Run the simulation.
    sol = simulate(UnforcedProblem(p, ω), T ; kwargs...)

    # --> Animate the dipole.
    x, y = mesh(p)

    anim = @animate for i = 1:length(sol.t)
        plot_vorticity(sol[i], p)
    end

    gif(anim, "../imgs/dipole.gif", fps=25)

    return
end

###################################
#####                         #####
#####     KOLMOGOROV FLOW     #####
#####                         #####
###################################

mutable struct KolmogorovFlow{T} <: NS2DProblem{T}

    # --> Unforced dynamics.
    unforced_dyn::UnforcedProblem{T}

    # --> Steady forcing.
    f::Array{Complex{T}, 2}

end

function KolmogorovFlow(p::SimParams, ω::Array{Complex{T}, 2}, n::Int) where T <: Real

    # --> Extract the parameters of the simulation.
    @unpack Lx, Ly, nx, ny, ν = p

    # --> Unforced part of the dynamics.
    unforced_dyn = UnforcedProblem(p, ω)

    # --> Kolmogorov flow forcing.
    x, y = mesh(p)
    x, y = collect(x), collect(y)

    f = -n .* cos.(n .* y * one.(x)')
    f = fft(f)

    # --> Return the container.
    prob = KolmogorovFlow(unforced_dyn, f)

    return prob
end

function rhs!(dΩ::Array{Complex{T},2}, Ω::Array{Complex{T},2}, p::KolmogorovFlow{T}, t::T) where T <: Real

    # --> Unforced dynamics.
    rhs!(dΩ, Ω, p.unforced_dyn, t)

    # --> Add forcing.
    dΩ .+= p.f

    return
end


function simulate(nsprob::KolmogorovFlow, T ; alg=Tsit5(), kwargs...)

    # --> Type of numbers used.
    N = findparams(nsprob)

    # --> Set the ODE Problem.
    tspan = (zero(N), convert(N, T))
    prob = ODEProblem(rhs!, copy(nsprob.unforced_dyn.ω₀), tspan, nsprob)

    # --> Integrate forward in time.
    sol = solve(prob, alg ; kwargs...)

    return sol
end

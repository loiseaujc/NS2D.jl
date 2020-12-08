using NS2D
using Statistics
using FFTW

function main(; n=256, ν=1e-3, T=10)

    # --> Parameters for the simulation.
    p = SimParams(n=n, ν=ν)

    # --> Taylor-Green initial condition.
    ω₀ = taylor_green(p)

    # --> Time-dependent analytic solution.
    ground_truth(t) = ω₀ .* exp(-2ν*t)

    # --> Simulate using NS2D
    sol = simulate(ω₀, p, T)

    # --> Extract the final time.
    ω = sol[2]

    # --> Compute the L2 and Linf norm.
    ℓ₂ = mean( real(ifft(ground_truth(T)-ω)).^2 )
    @show ℓ₂

    ℓ∞ = maximum( abs.(ifft(ground_truth(T)-ω)) )
    @show ℓ∞

    return
end

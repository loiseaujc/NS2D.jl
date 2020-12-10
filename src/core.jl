abstract type NS2DProblem{T<:Real} end

findparams(::NS2DProblem{T}) where T = T

"""
    SimParams(L=2π, n=256, ν=1e-4)

Parameters of the simulation. L denotes the size of the doubly
periodic computational domain, n the number of grid points to
discretize it in both direction and ν is the kinematic viscosity
of the working fluid.
"""
@with_kw struct SimParams
    # --> Dimension of the domain.
    L::Float64 = 2π ; @assert L > 0

    # --> Number of grid points per direction.
    n::Int = 256 ; @assert mod(n, 2) == 0

    # --> Viscosity.
    ν::Float64 = 1e-4 ; @assert ν > 0
end

mutable struct UnforcedProblem{T} <: NS2DProblem{T}
    # --> Number of grid points.
    n::Int64

    # --> Viscosity.
    ν::T

    # --> Wavenumbers
    α::Array{T,1}
    β::Array{T,1}

    # --> Initial condition.
    ω₀::Array{Complex{T}, 2}

    # --> Workiing arrays for the computation of the advection term.
    adv::Array{Complex{T},2}
    ∂ω∂x::Array{Complex{T},2}
    ∂ω∂y::Array{Complex{T},2}
    u::Array{Complex{T},2}
    v::Array{Complex{T},2}

    # --> Working arrays for the computation of the dealiased advection term.
    advp::Array{Complex{T},2}
    ∂ω∂xp::Array{Complex{T},2}
    ∂ω∂yp::Array{Complex{T},2}
    up::Array{Complex{T},2}
    vp::Array{Complex{T},2}
end

function UnforcedProblem(p::SimParams, x::Array{Complex{T}, 2}) where T <: Real

    # --> Extract the parameters of the simulation.
    @unpack L, n, ν = p

    # --> Compute the wavenumbers.
    α = fftfreq(n, n/L) * 2π

    # --> Convert ν and α to whatever type T is.
    ν = convert(T, ν)
    α = convert(Array{T, 1}, α)

    # --> Template for the padded arrays.
    y = zeros(eltype(x), (3n÷2, 3n÷2))

    # --> Return the container.
    prob = UnforcedProblem(
        n, ν, α, α, x,
        zero(x), zero(x), zero(x), zero(x), zero(x),
        zero(y), zero(y), zero(y), zero(y), zero(y)
    )

    return prob
end


function simulate(nsprob::UnforcedProblem, T ; alg=Tsit5(), kwargs...)

    # --> Set the ODE Problem.
    tspan = (zero(findparams(nsprob)), convert(findparams(nsprob), T))
    prob = ODEProblem(unforced_dynamics!, copy(nsprob.ω₀), tspan, nsprob)

    # --> Integrate forward in time.
    sol = solve(prob, alg ; kwargs...)

    return sol
end

function unforced_dynamics!(dΩ::Array{Complex{T},2}, Ω::Array{Complex{T},2}, p::UnforcedProblem{T}, t::T) where T <: Real

    # --> Arrays for the computation of the advection term.
    adv, ∂ω∂x, ∂ω∂y, u, v = p.adv, p.∂ω∂x, p.∂ω∂y, p.u, p.v

    # --> Arrays for the dealiased computation.
    advp, ∂ω∂xp, ∂ω∂yp, up, vp = p.advp, p.∂ω∂xp, p.∂ω∂yp, p.up, p.vp

    # --> Compute all required quantities in spectral space.
    @inbounds for j = 1:p.n÷2
        @inbounds @simd for i = 1:p.n
            # --> Misc.
            ω, α, β = Ω[i, j], p.α[j], p.β[i]
            k² = α^2 + β^2

            # --> Solve for the streamfunction.
            ψ = i == 1 && j == 1 ? zero(ω) : ω/k²

            # --> Diffusion term.
            dΩ[i, j] = -p.ν * k² * ω

            # --> Unpadded gradient of the vorticity field in spectral space.
            ∂ω∂x[i, j] = im * α * ω
            ∂ω∂y[i, j] = im * β * ω

            # --> Unpadded velocity in spectral space.
            u[i, j] = im * β * ψ
            v[i, j] = -im * α * ψ
        end
    end

    # --> Enforce the Hermitian symmetry in the Fourier transform
    #     since the loop above only acted on half the frequency space.
    enforce_hermitian_symmetry!(dΩ)
    enforce_hermitian_symmetry!(∂ω∂x), enforce_hermitian_symmetry!(∂ω∂y)
    enforce_hermitian_symmetry!(u), enforce_hermitian_symmetry!(v)

    # --> Padding in spectral space.
    spectral_pad!(up, u), spectral_pad!(vp, v)
    spectral_pad!(∂ω∂xp, ∂ω∂x), spectral_pad!(∂ω∂yp, ∂ω∂y)

    # --> Compute the advection in physical space.
    ifft!(up), ifft!(vp), ifft!(∂ω∂xp), ifft!(∂ω∂yp)
    @. advp = up * ∂ω∂xp + vp * ∂ω∂yp

    # --> Chop the high-wavenumbers (anti-aliasing)
    spectral_chop!(adv, fft!(advp))

    # --> Compute the left-hand side.
    dΩ .-= adv
    dΩ[:, p.n÷2+1] = dΩ[p.n÷2+1, :] .= 0

    return
end

function enforce_hermitian_symmetry!(x)

    n = size(x, 1)

    x[1, end:-1:n÷2+2] = conj(@view x[1, 2:n÷2])
    x[2:end, end:-1:n÷2+2] = conj(@view x[end:-1:2, 2:n÷2])

    return
end

function spectral_pad!(y, x)

    n = size(x, 1)

    y[1:n÷2, 1:n÷2] = x[1:n÷2, 1:n÷2]
    y[1:n÷2, end-n÷2+1:end] = x[1:n÷2, n÷2+1:end]
    y[end-n÷2+1:end, 1:n÷2] = x[n÷2+1:end, 1:n÷2]
    y[end-n÷2+1:end, end-n÷2+1:end] = x[n÷2+1:end, n÷2+1:end]

    return
end

function spectral_chop!(y, x)

    n = size(y, 1)

    y[1:n÷2, 1:n÷2] = x[1:n÷2, 1:n÷2]
    y[1:n÷2, n÷2+1:end] = x[1:n÷2, end-n÷2+1:end]
    y[n÷2+1:end, 1:n÷2] = x[end-n÷2+1:end, 1:n÷2]
    y[n÷2+1:end, n÷2+1:end] = x[end-n÷2+1:end, end-n÷2+1:end]

    return
end

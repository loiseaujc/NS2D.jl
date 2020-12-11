abstract type NS2DProblem{T<:Real} end

findparams(::NS2DProblem{T}) where T = T

"""
    SimParams(Lx=2π, Ly=2π, nx=256, ny=256, ν=1e-4)

Parameters of the simulation. L denotes the size of the doubly
periodic computational domain, n the number of grid points to
discretize it in both direction and ν is the kinematic viscosity
of the working fluid.
"""
@with_kw struct SimParams
    # --> Dimension of the domain in the x-direction.
    Lx::Float64 = 2π ; @assert Lx > 0

    # --> Dimension of the domain in the y-direction.
    Ly::Float64 = 2π ; @assert Ly > 0

    # --> Number of grid points in the x-direction.
    nx::Int = 256 ; @assert mod(nx, 2) == 0

    # --> Number of grid points in the y-direction.
    ny::Int = 256 ; @assert mod(ny, 2) == 0

    # --> Viscosity.
    ν::Float64 = 1e-4 ; @assert ν > 0
end

#####################################
#####                           #####
#####     UNFORCED DYNAMICS     #####
#####                           #####
#####################################

mutable struct UnforcedProblem{T} <: NS2DProblem{T}
    # --> Number of grid points in the x-direction.
    nx::Int64

    # --> Number of grid points in the x-direction.
    ny::Int64

    # --> Viscosity.
    ν::T

    # --> Wavenumbers
    α::Array{T,1}
    β::Array{T,1}

    # --> Initial condition.
    ω₀::Array{Complex{T}, 2}

    # --> Working arrays for the computation of the advection term.
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
    @unpack Lx, Ly, nx, ny, ν = p

    # --> Compute the wavenumbers.
    α = fftfreq(nx, nx/Lx) * 2π
    β = fftfreq(ny, ny/Ly) * 2π

    # --> Convert ν and α to whatever type T is.
    ν = convert(T, ν)
    α = convert(Array{T, 1}, α)
    β = convert(Array{T, 1}, β)

    # --> Template for the padded arrays.
    y = zeros(eltype(x), (3ny÷2, 3nx÷2))

    # --> Return the container.
    prob = UnforcedProblem(
        nx, ny, ν, α, β, x,
        zero(x), zero(x), zero(x), zero(x), zero(x),
        zero(y), zero(y), zero(y), zero(y), zero(y)
    )

    return prob
end

function unforced_dynamics!(dΩ::Array{Complex{T},2}, Ω::Array{Complex{T},2}, p::UnforcedProblem{T}, t::T) where T <: Real

    # --> Arrays for the computation of the advection term.
    adv, ∂ω∂x, ∂ω∂y, u, v = p.adv, p.∂ω∂x, p.∂ω∂y, p.u, p.v

    # --> Arrays for the dealiased computation.
    advp, ∂ω∂xp, ∂ω∂yp, up, vp = p.advp, p.∂ω∂xp, p.∂ω∂yp, p.up, p.vp

    # --> Compute all required quantities in spectral space.
    @inbounds for j = 1:p.nx
        @inbounds @simd for i = 1:p.ny
            # --> Misc.
            ω, α, β = Ω[i, j], p.α[j], p.β[i]
            k² = α^2 + β^2

            # --> Solve for the streamfunction.
            ψ = i == j == 1 ? zero(ω) : ω/k²

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

    # # --> Enforce the Hermitian symmetry in the Fourier transform
    # #     since the loop above only acted on half the frequency space.
    # enforce_hermitian_symmetry!(dΩ)
    # enforce_hermitian_symmetry!(∂ω∂x), enforce_hermitian_symmetry!(∂ω∂y)
    # enforce_hermitian_symmetry!(u), enforce_hermitian_symmetry!(v)

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
    dΩ[:, p.nx÷2+1] .= 0
    dΩ[p.ny÷2+1, :] .= 0

    return
end

function enforce_hermitian_symmetry!(x)

    n = size(x, 1)

    x[1, end:-1:n÷2+2] = conj(@view x[1, 2:n÷2])
    x[2:end, end:-1:n÷2+2] = conj(@view x[end:-1:2, 2:n÷2])

    return
end

function spectral_pad!(y, x)

    ny, nx = size(x)

    y[1:ny÷2, 1:nx÷2] = x[1:ny÷2, 1:nx÷2]
    y[1:ny÷2, end-nx÷2+1:end] = x[1:ny÷2, nx÷2+1:end]
    y[end-ny÷2+1:end, 1:nx÷2] = x[ny÷2+1:end, 1:nx÷2]
    y[end-ny÷2+1:end, end-nx÷2+1:end] = x[ny÷2+1:end, nx÷2+1:end]

    return
end

function spectral_chop!(y, x)

    ny, nx = size(y)

    y[1:ny÷2, 1:nx÷2] = x[1:ny÷2, 1:nx÷2]
    y[1:ny÷2, nx÷2+1:end] = x[1:ny÷2, end-nx÷2+1:end]
    y[ny÷2+1:end, 1:nx÷2] = x[end-ny÷2+1:end, 1:nx÷2]
    y[ny÷2+1:end, nx÷2+1:end] = x[end-ny÷2+1:end, end-nx÷2+1:end]

    return
end

rhs!(dΩ::Array{Complex{T},2}, Ω::Array{Complex{T},2}, p::UnforcedProblem{T}, t::T) where T <: Real = unforced_dynamics!(dΩ::Array{Complex{T},2}, Ω::Array{Complex{T},2}, p::UnforcedProblem{T}, t::T)

function simulate(nsprob::UnforcedProblem, T ; alg=Tsit5(), kwargs...)

    # --> Type of numbers used.
    N = findparams(nsprob)

    # --> Set the ODE Problem.
    tspan = (zero(N), convert(N, T))
    prob = ODEProblem(rhs!, copy(nsprob.ω₀), tspan, nsprob)

    # --> Integrate forward in time.
    sol = solve(prob, alg ; kwargs...)

    return sol
end

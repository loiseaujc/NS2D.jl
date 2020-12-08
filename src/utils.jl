function compute_reynolds_number(ω, p)

     # --> Compute the velocity.
     vx, vy = compute_velocity(ω, p)

     # --> Mean-squared amplitude.
     E = mean(u.^2 + v.^2)

     # --> Reynolds number.
     Re = √E * p.L / p.ν

    return Re
end

function compute_velocity(ω, p)

    # --> Extract parameters.
    L, n = p.L, p.n

    # --> Get the wavenumbers.
    α = β = fftfreq(n, n/L) * 2π

    # --> Initialize arrays.
    vx, vy = zero(ω), zero(ω)

    # --> Compute the velocity component in spectral space.
    for j = 1:n, i = 1:n

        # --> Streamfunction.
        k² = α[j]^2 + β[i]^2
        ψ = i == j == 1 ? zero(ω[i, j]) : ω[i, j] / k²

        # --> Velocity.
        vx[i, j] = im * β[i] * ψ
        vy[i, j] = im * α[j] * ψ

    end

    # --> From spectral space to physical space.
    ifft!(vx), ifft!(vy)

    return real(vx), real(vy)
end

function compute_taylor_sale(ω, p)

    # --> Compute the velocity field.
    vx, vy = compute_velocity(ω, p)

    # --> Compute the amplitude of the velocity fluctuation.
    E = mean(u.^2 + v.^2)

    # --> Compute the amplitude of the vorticity fluctuation.
    Q = mean(real(ifft(ω)).^2)

    # --> Compute the Taylor micro-scale.
    λ = √(E/Q)

    return λ
end

function mesh(p)

    # --> Unpack the paramters.
    @unpack L, n = p

    # --> Construct the one-dimensional meshes.
    x = y = LinRange(-L/2, L/2, n+1)[1:end-1]

    return x, y
end

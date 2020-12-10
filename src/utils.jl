function compute_reynolds_number(ω, p)

     # --> Compute the velocity.
     vx, vy = compute_velocity(ω, p)

     # --> Mean-squared amplitude.
     E = mean(vx.^2 + vy.^2)

     # --> Reynolds number.
     Re = √E * p.Lx / p.ν

    return Re
end

function compute_velocity(ω, p)

    # --> Extract parameters.
    @unpack Lx, Ly, nx, ny = p

    # --> Get the wavenumbers.
    α = fftfreq(nx, nx/Lx) * 2π
    β = fftfreq(ny, ny/Ly) * 2π

    # --> Initialize arrays.
    vx, vy = zero(ω), zero(ω)

    # --> Compute the velocity component in spectral space.
    for j = 1:nx, i = 1:ny

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

function compute_taylor_scale(ω, p)

    # --> Compute the velocity field.
    vx, vy = compute_velocity(ω, p)

    # --> Compute the amplitude of the velocity fluctuation.
    E = mean(vx.^2 + vy.^2)

    # --> Compute the amplitude of the vorticity fluctuation.
    Q = mean(real(ifft(ω)).^2)

    # --> Compute the Taylor micro-scale.
    λ = √(E/Q)

    return λ
end

function mesh(p)

    # --> Unpack the paramters.
    @unpack Lx, Ly, nx, ny = p

    # --> Construct the one-dimensional meshes.
    x = LinRange(-Lx/2, Lx/2, nx+1)[1:end-1]
    y = LinRange(-Ly/2, Ly/2, ny+1)[1:end-1]

    return x, y
end

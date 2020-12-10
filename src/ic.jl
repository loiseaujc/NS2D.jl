function isotropic_turbulence(spectrum, p)

    # --> Get the simulation parameters.
    @unpack Lx, Ly, nx, ny = p

    # --> Wavenumbers.
    α = fftfreq(nx, nx/Lx) * 2π
    β = fftfreq(ny, ny/Ly) * 2π

    # --> Random Fourier modes with unit energy.
    ω = randn(ny, nx)
    ω .-= mean(ω)

    ω = fft(ω)
    ω ./= abs.(ω)

    # --> Apply the isotropic enstrophy spectrum
    for j = 1:nx, i = 1:ny
        # --> Wavenumber
        k = √(α[j]^2 + β[i]^2)

        # --> Rescale the Fourier mode.
        ω[i, j] = i == j == 1 ? zero(ω[i, j]) : ω[i, j] * √spectrum(k)
    end

    # --> Zero-out the last wavenumber.
    ω[:, nx÷2+1] .= 0
    ω[ny÷2+1, :] .= 0

    # --> Scale the vorticity field to the desired Reynolds number.
    ω = scale_by_re(ω, p)

    return ω
end

function scale_by_re(ω, p)

    # --> Simulation parameters.
    @unpack Lx, Ly, nx, ny = p

    # --> Rescale the vorticity field to enforce the desired Re.
    u, v = compute_velocity(ω, p)

    E = mean(u.^2 + v.^2)

    ω /= √E * 2π

    return ω
end

function isolated_vortex(x, y ; x₀=0.0, y₀=0.0, Γ=1.0, r=1.0)

    # --> Center the mesh around the (x₀, y₀) position.
    Lx = maximum(abs.(x))
    Ly = maximum(abs.(y))
    x, y = collect(x .- x₀), collect(y .- y₀)

    # --> Enforce periodicity.
    x[x .> Lx] .= x[x .> Lx] .- 2Lx
    x[x .< -Lx] .= x[x .< -Lx] .+ 2Lx

    y[y .> Ly] .= y[y .> Ly] .- 2Ly
    y[y .< -Ly] .= y[y .< -Ly] .+ 2Ly

    # --> Isolated vortex vorticity field.
    r² = (x').^2 .+ y.^2
    ω = Γ/(π*r^2) .* exp.(-r²./r^2)

    # --> Map to spectral space.
    ω = fft(ω)

    return ω
end

function dipole(p)

    # --> Unpack parameters.
    @unpack Lx, Ly, nx, ny = p

    # --> Get mesh.
    x, y = mesh(p)

    # --> Dipole.
    ω = isolated_vortex(x, y ; Γ=1.0, r=0.5, y₀=0.5)
    ω += isolated_vortex(x, y ; Γ=-1.0, r=0.5, y₀=-0.5)

    # --> Remove the mean.
    ω[1, 1] = 0.0

    # --> Scale vorticity field to the desired Reynolds number.
    ω = scale_by_re(ω, p)

    return ω
end

function taylor_green(p)

    # --> Construct the mesh.
    x, y = mesh(p)

    # --> Taylor-Green vortices.
    ω = -2 .* cos.(x') .* cos.(y)

    # --> Map to spectral space.
    ω = fft(ω)

    return ω
end

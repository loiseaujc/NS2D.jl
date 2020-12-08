function isotropic_turbulence(spectrum, p)

    # --> Get the simulation parameters.
    L, n, ν = p.L, p.n, p.ν

    # --> Wavenumbers.
    α = β = fftfreq(n, n/L) * 2π

    # --> Random Fourier modes with unit energy.
    ω = randn(n, n)
    ω .-= mean(ω)

    ω = fft(ω)
    ω ./= abs.(ω)

    # --> Apply the isotropic enstrophy spectrum
    for j = 1:n, i = 1:n
        # --> Wavenumber
        k = √(α[j]^2 + β[i]^2)

        # --> Rescale the Fourier mode.
        ω[i, j] = i == j == 1 ? zero(ω[i, j]) : ω[i, j] * √spectrum(k)
    end

    # --> Zero-out the last wavenumber.
    ω[:, n÷2+1] = ω[n÷2+1, :] .= 0

    # --> Scale the vorticity field to the desired Reynolds number.
    ω = scale_by_re(ω, p)

    return ω
end

function scale_by_re(ω, p)

    # --> Simulation parameters.
    L, n, ν = p.L, p.n, p.ν

    # --> Rescale the vorticity field to enforce the desired Re.
    u, v = compute_velocity(ω, p)

    E = mean(u.^2 + v.^2)

    ω /= √E * 2π

    return ω
end

function isolated_vortex(x, y ; x₀=0.0, y₀=0.0, Γ=1.0, r=1.0)

    # --> Center the mesh around the (x₀, y₀) position.
    L = maximum(abs.(x))
    x, y = collect(x .- x₀), collect(y .- y₀)

    # --> Enforce periodicity.
    x[x .> L] .= x[x .> L] .- 2L
    x[x .< -L] .= x[x .< -L] .+ 2L

    y[y .> L] .= y[y .> L] .- 2L
    y[y .< -L] .= y[y .< -L] .+ 2L

    # --> Isolated vortex vorticity field.
    r² = (x').^2 .+ y.^2
    ω = Γ/(π*r^2) .* exp.(-r²./r^2)

    return ω
end

function dipole(p)

    # --> Unpack parameters.
    @unpack L, n, ν = p

    # --> Get mesh.
    x, y = mesh(p)

    # --> Dipole.
    ω = isolated_vortex(x, y ; Γ=1.0, r=0.5, y₀=0.5)
    ω += isolated_vortex(x, y ; Γ=-1.0, r=0.5, y₀=-0.5)

    # --> Remove the mean and map to spectral space.
    ω = fft(ω .- mean(ω))

    # --> Scale vorticity field to the desired Reynolds number.
    ω = scale_by_re(ω, p)

    return ω
end

function taylor_green(x, y)

    # --> Taylor-Green vortices.
    ω = -2 .* cos.(x') .* cos.(y)

    # --> Map to spectral space.
    ω = fft(ω)

    return ω
end

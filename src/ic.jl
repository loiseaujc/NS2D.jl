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

#!/usr/bin/env python3
"""
make_ico_main_figure.py

Main-text ICO figure generator (quantum-switch "Jones microscope").
Builds a physically meaningful Jones sample with commuting/non-commuting regions,
simulates single-run counts for the {+,−} control readout, and renders a 2×2 figure:

(1) Sample parameters (Δ, anisotropy, β) and ||[U,V]||_F ground-truth map
(2) Balanced commutator-contrast Ĉ(x,y) = N_− / (N_+ + N_−)
(3) Per-pixel standard error σ_Ĉ from binomial statistics
(4) A small analyzer-independence check (printed) for (N_+ + N_−)

Usage (defaults are sensible for the paper):
    python3 make_ico_main_figure.py --H 256 --W 256 --N 800 --nu 0.9 --out ico_main_figure.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------- Jones optics primitives -------------------------------

def rot(theta):
    """
    Real 2×2 rotation acting on Jones vectors (passive rotation of axes).
    Convention: R(theta) = [[cos, -sin],[sin, cos]].
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    return R


def retarder(alpha, delta):
    """
    Linear retarder with fast-axis at angle alpha and retardance delta (radians).
    J_R(alpha,delta) = R(-alpha) @ diag(e^{iΔ/2}, e^{-iΔ/2}) @ R(alpha)
    Returns complex 2x2 matrix.
    """
    Ra  = rot(alpha)
    Rmt = rot(-alpha)
    phase = np.array([[np.exp(1j*delta/2.0), 0.0],
                      [0.0, np.exp(-1j*delta/2.0)]], dtype=complex)
    return Rmt @ phase @ Ra


def diattenuator(beta, aniso):
    """
    Partial linear polarizer (diattenuator) oriented at beta (radians).
    aniso in [0,1): transmitted intensities along principal axes are
      t_H = 1,  t_V = 1 - aniso
    Jones magnitudes are sqrt of intensities.

    J_D(beta,aniso) = R(-beta) @ diag( sqrt(t_H), sqrt(t_V) ) @ R(beta)
    Returns complex (real-valued) 2x2 matrix.
    """
    tH = 1.0
    tV = 1.0 - np.clip(aniso, 0.0, 0.999999)
    Ra  = rot(beta)
    Rmt = rot(-beta)
    Dmag = np.array([[np.sqrt(tH), 0.0],
                     [0.0, np.sqrt(tV)]], dtype=float)
    return Rmt @ Dmag @ Ra


def jones_commutator_norm(U, V):
    """ Frobenius norm of the commutator ||[U,V]||_F for 2x2 complex matrices. """
    C = V @ U - U @ V
    return np.sqrt(np.real(np.trace(C.conj().T @ C)))


# ------------------------------- Simulation of the quantum switch -------------------------------

def switch_intensities(U, V, psi_in, nu=1.0):
    """
    Compute I_+, I_- for the quantum switch with imperfect indistinguishability nu ∈ [0,1].
    Ideal: |ψ±> ∝ (UV ± VU)|ψ_in>,  I± = ||ψ±||^2
    Imperfect: damp the interference term via nu:
        I± = ||UVψ||^2 + ||VUψ||^2  ±  2*nu*Re( <UVψ | VUψ> )
    This preserves I_+ + I_- = 2( ||UVψ||^2 + ||VUψ||^2 ) (analyzer-independent sum).
    """
    j1 = U @ (V @ psi_in)    # UV|ψ>
    j2 = V @ (U @ psi_in)    # VU|ψ>
    n1 = np.vdot(j1, j1).real
    n2 = np.vdot(j2, j2).real
    cross = (np.vdot(j1, j2)).real
    I_plus  = n1 + n2 + 2.0 * nu * cross
    I_minus = n1 + n2 - 2.0 * nu * cross
    # Numerical safety
    I_plus  = max(I_plus,  0.0)
    I_minus = max(I_minus, 0.0)
    return I_plus, I_minus, n1, n2


def simulate_counts(Ip, Im, N, rng):
    """
    Given expected intensities {Ip, Im} per pixel (nonnegative), draw
    total events N (Poisson or fixed) and split with a binomial trial.
    Here we take a fixed N per pixel for clarity (can be Poisson if desired).
    """
    s = Ip + Im
    if s <= 0:
        return 0, 0, 0, 0.0
    p_minus = Im / s
    # Binomial draw for N−; the rest goes to N+
    Nm = rng.binomial(N, p_minus)
    Np = N - Nm
    # Balanced estimator Ĉ = Nm / (Np+Nm); SE from binomial
    C_hat = Nm / max(N, 1)
    se = np.sqrt(p_minus * (1.0 - p_minus) / max(N, 1))
    return Np, Nm, C_hat, se


# ------------------------------- Scene builder -------------------------------

def build_scene(H, W,
                delta_commuting=0.0,     # Δ in commuting region (e.g., 0)
                delta_nonc=np.pi/2,      # Δ in non-commuting region (e.g., quarter-wave)
                aniso_commuting=0.0,     # no diattenuation in commuting region
                aniso_nonc=0.6,          # strong diattenuation in non-commuting region
                beta_commuting=0.0,      # axes aligned in commuting region
                beta_nonc=np.pi/4,       # axis mismatch in non-commuting region
                alpha_all=0.0            # retarder fast-axis orientation everywhere
                ):
    """
    Construct per-pixel U (retarder) and V (diattenuator) fields with a clear
    commuting vs non-commuting segmentation:

      Left half: commuting (β=0, aniso≈0 → [U,V]≈0 regardless of Δ).
      Right half: non-commuting (β=β_nonc, aniso>0, Δ>0 → [U,V]≠0).

    Returns:
      U_map, V_map : arrays of shape (H,W,2,2) complex
      fields       : dict of per-pixel parameter maps (Δ, aniso, β, ||[U,V]||_F)
    """
    U_map = np.zeros((H, W, 2, 2), dtype=complex)
    V_map = np.zeros((H, W, 2, 2), dtype=complex)

    # Fields for plotting
    delta_field  = np.zeros((H, W), dtype=float)
    aniso_field  = np.zeros((H, W), dtype=float)
    beta_field   = np.zeros((H, W), dtype=float)
    comm_norm    = np.zeros((H, W), dtype=float)

    # Region masks
    mask_nonc = np.zeros((H, W), dtype=bool)
    mask_nonc[:, W//2:] = True   # right half non-commuting

    # Fill fields
    for y in range(H):
        for x in range(W):
            if mask_nonc[y, x]:
                delta  = delta_nonc
                aniso  = aniso_nonc
                beta   = beta_nonc
            else:
                delta  = delta_commuting
                aniso  = aniso_commuting
                beta   = beta_commuting

            U = retarder(alpha_all, delta)      # U: linear retarder
            V = diattenuator(beta, aniso)       # V: partial polarizer

            U_map[y, x] = U
            V_map[y, x] = V

            delta_field[y, x] = delta
            aniso_field[y, x] = aniso
            beta_field[y, x]  = beta
            comm_norm[y, x]   = jones_commutator_norm(U, V)

    fields = dict(delta=delta_field, aniso=aniso_field, beta=beta_field, comm=comm_norm)
    return U_map, V_map, fields, mask_nonc


# ------------------------------- Figure generator -------------------------------

def render_main_figure(fields, C_hat, SE, out_png):
    """
    Produce the 2×2 main-text figure:
      (a) Parameter maps: Δ, anisotropy, β + overlay of commuting vs non-commuting
      (b) Ground-truth ||[U,V]||_F
      (c) Balanced commutator-contrast Ĉ
      (d) Per-pixel standard error σ_Ĉ
    """
    H, W = fields['delta'].shape
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 8.2))
    plt.subplots_adjust(wspace=0.18, hspace=0.22)

    # Panel (a): sample parameters as a stitched RGB-like visualization
    # Red channel ~ Δ/π, Green ~ anisotropy, Blue ~ |β|/(π/2)
    # This is only a compact way to show where parameters differ.
    comp = np.zeros((H, W, 3), dtype=float)
    comp[..., 0] = np.clip(fields['delta'] / np.pi, 0, 1)
    comp[..., 1] = np.clip(fields['aniso'] / 0.9,  0, 1)
    comp[..., 2] = np.clip(np.abs(fields['beta']) / (np.pi/2), 0, 1)

    ax = axs[0, 0]
    ax.imshow(comp, origin='lower', interpolation='nearest')
    ax.set_title('Sample parameters: Δ (R), anisotropy (G), β (B)')
    ax.set_xticks([]); ax.set_yticks([])

    # Panel (b): Frobenius norm of commutator
    ax = axs[0, 1]
    im = ax.imshow(fields['comm'], origin='lower', interpolation='nearest')
    ax.set_title(r'Ground truth  $\|[U,V]\|_F$')
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\|[U,V]\|_F$')

    # Panel (c): Ĉ map
    ax = axs[1, 0]
    im = ax.imshow(C_hat, origin='lower', vmin=0.0, vmax=1.0, interpolation='nearest')
    ax.set_title(r'Balanced commutator contrast  $\hat C(x,y)=\frac{N_-}{N_+ + N_-}$')
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\hat C$')

    # Panel (d): σ_Ĉ map
    ax = axs[1, 1]
    vmax = np.percentile(SE, 99.5)
    im = ax.imshow(SE, origin='lower', vmin=0.0, vmax=vmax, interpolation='nearest')
    ax.set_title(r'Per-pixel standard error  $\sigma_{\hat C}$  (binomial)')
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\sigma_{\hat C}$')

    fig.suptitle('Quantum-switch Jones microscopy: commutator contrast from one labeled dataset', y=0.995)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved figure: {out_png}")


# ------------------------------- Main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Main-text ICO figure generator (quantum-switch Jones microscopy).")
    ap.add_argument('--H', type=int, default=256, help='image height (pixels)')
    ap.add_argument('--W', type=int, default=256, help='image width (pixels)')
    ap.add_argument('--N', type=int, default=800, help='coincidences per pixel devoted to ICO (S_ICO)')
    ap.add_argument('--nu', type=float, default=0.95, help='indistinguishability/visibility (0..1) for the switch')
    ap.add_argument('--psi', type=float, default=np.pi/6, help='input polarization angle ψ (radians)')
    ap.add_argument('--xi', type=float, default=0.0, help='relative phase ξ of input Jones vector (radians)')
    ap.add_argument('--out', type=str, default='ico_main_figure.png', help='output PNG filename')
    ap.add_argument('--seed', type=int, default=7, help='rng seed')

    # Optional scene controls
    ap.add_argument('--delta-comm', type=float, default=0.0, help='Δ in commuting region')
    ap.add_argument('--delta-nonc', type=float, default=np.pi/2, help='Δ in non-commuting region')
    ap.add_argument('--aniso-comm', type=float, default=0.0, help='anisotropy in commuting region')
    ap.add_argument('--aniso-nonc', type=float, default=0.6, help='anisotropy in non-commuting region')
    ap.add_argument('--beta-comm', type=float, default=0.0, help='β in commuting region')
    ap.add_argument('--beta-nonc', type=float, default=np.pi/4, help='β in non-commuting region')
    ap.add_argument('--alpha', type=float, default=0.0, help='retarder axis α everywhere')

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # Build scene
    U_map, V_map, fields, mask_nonc = build_scene(
        H=args.H, W=args.W,
        delta_commuting=args.delta_comm,
        delta_nonc=args.delta_nonc,
        aniso_commuting=args.aniso_comm,
        aniso_nonc=args.aniso_nonc,
        beta_commuting=args.beta_comm,
        beta_nonc=args.beta_nonc,
        alpha_all=args.alpha
    )

    # Input Jones vector |ψ_in> = [cos ψ, e^{iξ} sin ψ]^T
    psi = args.psi
    xi  = args.xi
    psi_in = np.array([np.cos(psi), np.exp(1j*xi) * np.sin(psi)], dtype=complex)

    # Simulate counts and estimators
    H, W = args.H, args.W
    C_hat = np.zeros((H, W), dtype=float)
    SE    = np.zeros((H, W), dtype=float)
    sum_check = []

    for y in range(H):
        for x in range(W):
            U = U_map[y, x]
            V = V_map[y, x]
            I_plus, I_minus, n1, n2 = switch_intensities(U, V, psi_in, nu=args.nu)
            # Analyzer-independent sum check
            sum_check.append(I_plus + I_minus)
            # Draw counts and estimator
            Np, Nm, C_est, se = simulate_counts(I_plus, I_minus, args.N, rng)
            C_hat[y, x] = C_est
            SE[y, x] = se

    # Report completeness/analyzer-independence numerically
    sum_check = np.array(sum_check)
    rel_spread = (np.max(sum_check) - np.min(sum_check)) / max(np.mean(sum_check), 1e-12)
    print(f"[Check] Analyzer-independent sum I_+ + I_- relative spread across field: {rel_spread:.3e}")

    # Render figure
    render_main_figure(fields, C_hat, SE, args.out)


if __name__ == "__main__":
    main()

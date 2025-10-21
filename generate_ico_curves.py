#!/usr/bin/env python3
"""
Generate three ICO tuning curves for the paper:
1) Mean commutator contrast Ĉ vs anisotropy (ico_aniso)
2) Mean commutator contrast Ĉ vs retardance Δ (ico_delta)
3) Mean commutator contrast Ĉ vs axis mismatch β (ico_beta_const)

Uses the ICO module in qei_monte_carlo_gpu.py. Saves a PNG with 3 subplots,
and optionally a CSV with the sampled values.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Optional

import qei_monte_carlo_gpu as qei


def build_sim(H: int, W: int, ppp: int, gpu: bool,
              image: Optional[str], phi_mode: str, phi_image: Optional[str],
              T_const: Optional[float], phi_step_thresh: float, seed: int) -> qei.QEISimulator:
    sim = qei.QEISimulator(image_size=(H, W), photons_per_pixel=ppp, use_gpu=gpu, seed=seed)
    sim.load_sample(image_path=image,
                    phi_mode=phi_mode,
                    phi_image=phi_image,
                    T_const=T_const,
                    phi_step_thresh=phi_step_thresh)
    return sim


def mean_contrast(sim: qei.QEISimulator,
                  alpha_mode: str, alpha_const: float,
                  beta_mode: str, beta_const: float,
                  delta: float, aniso: float,
                  vin_psi: float, vin_xi: float) -> float:
    vin = (np.cos(vin_psi), np.exp(1j*vin_xi)*np.sin(vin_psi))
    sim.run_ico_acquisition(alpha_mode=alpha_mode,
                            alpha_const=alpha_const,
                            beta_mode=beta_mode,
                            beta_const=beta_const,
                            delta=delta,
                            aniso=aniso,
                            vin=vin)
    C_bal, _ = qei.QEISimulator.reconstruct_commutator_contrast(sim.counts['ICO'], nu=1.0)
    # Use mean absolute to avoid sign ambiguities from the balanced estimator
    return float(np.mean(np.abs(C_bal)))


def main():
    ap = argparse.ArgumentParser(description='Generate three ICO tuning curves (Ĉ vs aniso, Δ, β).')
    ap.add_argument('--H', type=int, default=128)
    ap.add_argument('--W', type=int, default=128)
    ap.add_argument('--ppp', type=int, default=1000, help='Photons per pixel for Monte Carlo')
    ap.add_argument('--gpu', action='store_true', help='Use GPU via CuPy if available')

    # Sample generation
    ap.add_argument('--image', type=str, default=None, help='Optional grayscale image for T')
    ap.add_argument('--T-const', type=float, default=None, help='Override T with a constant')
    ap.add_argument('--phi-mode', type=str, default='smooth', choices=['smooth','ramp','vortex','zernike','hue','gray','step'])
    ap.add_argument('--phi-image', type=str, default=None)
    ap.add_argument('--phi-step-thresh', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=12345)

    # ICO fixed parameters and sweeps
    ap.add_argument('--alpha-mode', type=str, default='const', choices=['from_phi','ramp','const'])
    ap.add_argument('--alpha-const', type=float, default=0.0)

    ap.add_argument('--npoints', type=int, default=21, help='Points per sweep')

    # Input polarization
    ap.add_argument('--vin-psi', type=float, default=np.pi/6, help='|ψ_in> orientation ψ (radians)')
    ap.add_argument('--vin-xi', type=float, default=0.3, help='|ψ_in> phase ξ (radians)')

    # Save
    ap.add_argument('--out', type=str, default='ico_curves.png', help='Output PNG filename')
    ap.add_argument('--csv', type=str, default=None, help='Optional CSV file to write sampled values')

    args = ap.parse_args()

    sim = build_sim(args.H, args.W, args.ppp, args.gpu,
                    args.image, args.phi_mode, args.phi_image,
                    args.T_const, args.phi_step_thresh, args.seed)

    # Curves setup
    aniso_vals = np.linspace(0.0, 0.9, args.npoints)
    delta_vals = np.linspace(0.0, np.pi, args.npoints)
    beta_vals  = np.linspace(0.0, np.pi/2, args.npoints)

    # Fixed values when sweeping others
    delta_fix = np.pi/2
    beta_fix  = np.pi/6
    aniso_fix = 0.6

    # Compute curves
    C_aniso = [mean_contrast(sim, args.alpha_mode, args.alpha_const,
                             'const', beta_fix, delta_fix, a,
                             args.vin_psi, args.vin_xi) for a in aniso_vals]

    C_delta = [mean_contrast(sim, args.alpha_mode, args.alpha_const,
                             'const', beta_fix, d, aniso_fix,
                             args.vin_psi, args.vin_xi) for d in delta_vals]

    C_beta  = [mean_contrast(sim, args.alpha_mode, args.alpha_const,
                             'const', b, delta_fix, aniso_fix,
                             args.vin_psi, args.vin_xi) for b in beta_vals]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    axs[0].plot(aniso_vals, C_aniso, 'o-')
    axs[0].set_xlabel('Anisotropy of V (ico_aniso)')
    axs[0].set_ylabel('Mean |Ĉ|')
    axs[0].set_title('Ĉ vs anisotropy')
    axs[0].grid(alpha=0.3)

    axs[1].plot(delta_vals, C_delta, 'o-')
    axs[1].set_xlabel('Retardance Δ (radians)')
    axs[1].set_ylabel('Mean |Ĉ|')
    axs[1].set_title('Ĉ vs Δ')
    axs[1].grid(alpha=0.3)

    axs[2].plot(beta_vals, C_beta, 'o-')
    axs[2].set_xlabel('Axis mismatch β (radians)')
    axs[2].set_ylabel('Mean |Ĉ|')
    axs[2].set_title('Ĉ vs β')
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(args.out, dpi=150)

    # CSV optional
    if args.csv is not None:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['param','value','mean_abs_C_hat'])
            for v, c in zip(aniso_vals, C_aniso):
                w.writerow(['aniso', v, c])
            for v, c in zip(delta_vals, C_delta):
                w.writerow(['delta', v, c])
            for v, c in zip(beta_vals, C_beta):
                w.writerow(['beta', v, c])

    print(f"Saved {args.out} (and {args.csv} if requested)")


if __name__ == '__main__':
    main()


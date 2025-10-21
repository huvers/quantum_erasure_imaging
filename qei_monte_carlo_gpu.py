
"""
Quantum Erasure Imaging (QEI): Correct Monte Carlo Simulator (GPU-accelerated)
-------------------------------------------------------------------------------
Implements the protocol and estimators as derived in the QEI manuscript:

• Correct joint four-way outcome intensities for HV, D/A, and rotated orthonormal analyzers
• Two-port balanced estimators for T and V_theta
• Phase retrieval (mod pi) from visibility using known amplitude A_theta(T)
• Optional two-step phase retrieval (global phase step δ) for unambiguous φ
• Vectorized per-pixel multinomial sampling:
    - CPU: NumPy with exact multinomial per pixel
    - GPU: CuPy with exact multinomial via sequential binomial decomposition (binomial thinning)

Usage (CLI examples)
--------------------
# CPU, synthetic sample, θ = π/8
python qei_monte_carlo_gpu_updated.py --H 128 --W 128 --ppp 1000 --theta 0.3926990817

# GPU (if CuPy+CUDA available), use hue of a color image as phase, and a grayscale for T
python qei_monte_carlo_gpu_updated.py --gpu --image amplitude.png --phi-mode hue --phi-image phase_color.png

# Two-step phase retrieval with a global step δ=π/3 at θ=π/4 (D/A)
python qei_monte_carlo_gpu_updated.py --ppp 2000 --theta 0.7853981634 --delta 1.0471975512

"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ---------------- GPU detection ----------------
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False

def asnumpy(x):
    if _CUPY_AVAILABLE and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x

# ---------------- Phase utility functions -----------
def hue_from_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """
    Vectorized RGB->[0,1) hue extraction (HSV model). img_rgb in [0,1], shape (H,W,3).
    Returns hue in [0,1).
    """
    r = img_rgb[...,0]
    g = img_rgb[...,1]
    b = img_rgb[...,2]
    maxc = np.maximum(np.maximum(r,g), b)
    minc = np.minimum(np.minimum(r,g), b)
    v = maxc
    delta = maxc - minc
    s = np.zeros_like(v)
    nonzero = maxc > 1e-12
    s[nonzero] = (delta[nonzero] / maxc[nonzero])
    hue = np.zeros_like(v)

    nz = delta > 1e-12
    rc = np.zeros_like(v); gc = np.zeros_like(v); bc = np.zeros_like(v)
    rc[nz] = (maxc[nz] - r[nz]) / delta[nz]
    gc[nz] = (maxc[nz] - g[nz]) / delta[nz]
    bc[nz] = (maxc[nz] - b[nz]) / delta[nz]

    mask_r = (nz) & (r == maxc)
    mask_g = (nz) & (g == maxc)
    mask_b = (nz) & (b == maxc)

    hue[mask_r] = (bc[mask_r] - gc[mask_r]) / 6.0
    hue[mask_g] = (2.0 + rc[mask_g] - bc[mask_g]) / 6.0
    hue[mask_b] = (4.0 + gc[mask_b] - rc[mask_b]) / 6.0

    hue = (hue % 1.0)
    return hue

def phase_from_hue(img_rgb: np.ndarray) -> np.ndarray:
    hue = hue_from_rgb(img_rgb)  # [0,1)
    return 2*np.pi*hue - np.pi   # [-pi, pi)

def phase_from_gray(img_gray: np.ndarray) -> np.ndarray:
    # img_gray in [0,1] -> phase in [-pi,pi)
    return np.pi*(2*img_gray - 1.0)

def phase_ramp(H: int, W: int, kx: float = 6*np.pi, ky: float = 4*np.pi) -> np.ndarray:
    y, x = np.indices((H,W))
    X = (x - (W-1)/2)/(W-1); Y = (y - (H-1)/2)/(H-1)
    phi = (kx*X + ky*Y) % (2*np.pi) - np.pi
    return phi

def phase_vortices(H: int, W: int, centers=[(0.3,0.3, +1), (0.7,0.6, -1)]) -> np.ndarray:
    y, x = np.indices((H,W))
    X = x/(W-1); Y = y/(H-1)
    phi = np.zeros((H,W), float)
    for cx, cy, m in centers:
        phi += m*np.arctan2(Y-cy, X-cx)
    return (phi + np.pi) % (2*np.pi) - np.pi

def phase_zernike_like(H: int, W: int, a: float = 1.0, b: float = 0.8, c: float = 0.6) -> np.ndarray:
    y, x = np.indices((H,W))
    yc, xc = (H-1)/2, (W-1)/2
    R = 0.48*min(H,W)
    Xn, Yn = (x-xc)/R, (y-yc)/R
    phi = a*(Xn**2 + Yn**2) + b*(Xn**2 - Yn**2) + c*(Xn*Yn)
    return (phi + np.pi) % (2*np.pi) - np.pi

# ---------------- Simulator ----------------
@dataclass
class QEIRunInfo:
    basis: str                 # 'HV', 'DA', or 'ROT'
    theta: Optional[float]     # radians for 'ROT', else None
    photons_per_pixel: int

class QEISimulator:
    """
    Quantum Erasure Imaging Monte Carlo (with GPU acceleration if available).

    Design notes
    ------------
    • Intensities are computed per pixel for the 4 joint outcomes:
        HV  : (D1,H), (D2,H), (D1,V), (D2,V)
        DA  : (D1,D), (D2,D), (D1,A), (D2,A)
        ROT : (D1,e1), (D2,e1), (D1,e2), (D2,e2) with e1/e2 orthonormal at angle θ
    • We sample exact multinomial counts per pixel.
      GPU path uses sequential binomial decomposition (Binomial thinning) which
      yields an exact Multinomial distribution while supporting elementwise p.
    • Two-port balanced estimators:
        T_hat = (n(D1,H)+n(D2,H))/(n(D1,V)+n(D2,V))
        V_DA  = ([D1,D]-[D1,A] - [D2,D]+[D2,A]) / sum_all
        V_ROT = ([D1,e1]-[D1,e2] - [D2,e1]+[D2,e2]) / sum_all
    """
    def __init__(self,
                 image_size: Tuple[int, int] = (128, 128),
                 photons_per_pixel: int = 1000,
                 use_gpu: bool = False,
                 seed: int = 12345):
        self.H, self.W = image_size
        self.photons_per_pixel = int(photons_per_pixel)
        self.use_gpu = bool(use_gpu and _CUPY_AVAILABLE)
        if self.use_gpu:
            cp.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.xp = cp if self.use_gpu else np
        self.T = None   # xp.ndarray of shape (H, W)
        self.phi = None # xp.ndarray of shape (H, W)
        self.counts: Dict[str, Dict[str, 'np.ndarray']] = {}

    # --------- sample (T, phi) generation ---------
    def load_sample(self,
                    image_path: Optional[str] = None,
                    phi_mode: str = "smooth",
                    phi_image: Optional[str] = None,
                    T_const: Optional[float] = None,
                    phi_step_thresh: float = 0.5) -> None:
        """
        Load amplitude T and construct phase φ.
        - image_path: grayscale image mapped to T in [0.1,0.9]; else synthetic T.
        - T_const: if provided, override T everywhere with this constant (clipped [0.1,0.9]).
        - phi_mode: 'smooth' (default), 'ramp', 'vortex', 'zernike', 'hue', 'gray', 'step'.
        - phi_image: if provided and phi_mode in {'hue','gray','step'}, file used to drive phase.
        - phi_step_thresh: threshold in [0,1] for 'step' mode.
        """
        H, W = self.H, self.W
        # --- T ---
        if image_path is not None:
            try:
                from PIL import Image
                img = Image.open(image_path).convert('L').resize((W, H))
                T = np.asarray(img, dtype=float) / 255.0
                T = np.clip(T, 0.1, 0.9)
            except Exception as e:
                print(f"[load_sample] Could not load amplitude image: {e}. Using synthetic T.")
                T = self._generate_test_T(H, W)
        else:
            T = self._generate_test_T(H, W)
        if T_const is not None:
            T = np.clip(float(T_const), 0.1, 0.99) * np.ones((H,W), float)

        # --- φ ---
        if phi_mode == "smooth":
            y = np.linspace(0, 2*np.pi, H)
            x = np.linspace(0, 2*np.pi, W)
            X, Y = np.meshgrid(x, y)
            phi = 0.5*np.sin(X) + 0.3*np.cos(Y) + 0.2*np.sin(X*Y/10.0)
        elif phi_mode == "ramp":
            phi = phase_ramp(H, W, kx=6*np.pi, ky=4*np.pi)
        elif phi_mode == "vortex":
            phi = phase_vortices(H, W)
        elif phi_mode == "zernike":
            phi = phase_zernike_like(H, W)
        elif phi_mode in ("hue","gray","step"):
            try:
                from PIL import Image
                if phi_mode == "hue":
                    img = np.asarray(Image.open(phi_image).convert('RGB').resize((W,H)), dtype=float)/255.0
                    phi = phase_from_hue(img)
                else:
                    img = np.asarray(Image.open(phi_image).convert('L').resize((W,H)), dtype=float)/255.0
                    if phi_mode == "gray":
                        phi = phase_from_gray(img)
                    else:
                        mask = (img > phi_step_thresh).astype(float)
                        phi = np.pi*(2*mask - 1.0)
            except Exception as e:
                print(f"[load_sample] Could not load phi image: {e}. Falling back to smooth φ.")
                y = np.linspace(0, 2*np.pi, H)
                x = np.linspace(0, 2*np.pi, W)
                X, Y = np.meshgrid(x, y)
                phi = 0.5*np.sin(X) + 0.3*np.cos(Y) + 0.2*np.sin(X*Y/10.0)
        else:
            raise ValueError(f"Unknown phi_mode: {phi_mode}")

        phi = (phi + np.pi) % (2*np.pi) - np.pi

        # Move to xp
        self.T = self.xp.asarray(T)
        self.phi = self.xp.asarray(phi)

    @staticmethod
    def _generate_test_T(H, W):
        yy, xx = np.indices((H, W))
        yc, xc = (H-1)/2.0, (W-1)/2.0
        r = np.hypot(yy - yc, xx - xc) / (0.48*min(H, W))
        T = 0.5 + 0.3*np.exp(-r**2) + 0.1*np.cos(4*np.pi*xx/W)*np.cos(4*np.pi*yy/H)
        return np.clip(T, 0.1, 0.9)

    # --------- joint intensities (unnormalized) ---------
    def _intensities_HV(self, T):
        # (D1,H), (D2,H), (D1,V), (D2,V)
        return (T/4, T/4, self.xp.ones_like(T)/4, self.xp.ones_like(T)/4)

    def _intensities_DA(self, T, phi):
        sT = self.xp.sqrt(T); cphi = self.xp.cos(phi)
        rD1D = T + 1 + 2*sT*cphi
        rD2D = T + 1 - 2*sT*cphi
        rD1A = T + 1 - 2*sT*cphi
        rD2A = T + 1 + 2*sT*cphi
        return (rD1D, rD2D, rD1A, rD2A)

    def _intensities_ROT(self, T, phi, theta: float):
        sT = self.xp.sqrt(T); cphi = self.xp.cos(phi)
        c, s = np.cos(theta), np.sin(theta)  # scalars in host; broadcast fine
        A = T*c*c + s*s
        B = T*s*s + c*c
        X = 2*sT*s*c*cphi
        # (D1,e1), (D2,e1), (D1,e2), (D2,e2)
        return (A+X, A-X, B-X, B+X)

    # --------- multinomial sampling (CPU/GPU) ---------
    def _normalize_four(self, a,b,c,d, eps=1e-18):
        # returns probabilities p1..p4 that sum to 1 elementwise
        tot = a + b + c + d
        tot = self.xp.where(tot <= 0, eps, tot)
        return a/tot, b/tot, c/tot, d/tot

    def _sample_fourway_cpu(self, M: int, p1, p2, p3, p4):
        """Exact per-pixel multinomial via NumPy (loop across pixels)."""
        H, W = p1.shape
        out = np.zeros((H, W, 4), dtype=np.int32)
        P = np.stack([asnumpy(p1), asnumpy(p2), asnumpy(p3), asnumpy(p4)], axis=-1)
        flat = P.reshape(-1, 4)
        for i in range(flat.shape[0]):
            out.reshape(-1,4)[i,:] = self.rng.multinomial(M, flat[i])
        return out[...,0], out[...,1], out[...,2], out[...,3]

    def _sample_fourway_gpu(self, M: int, p1, p2, p3, p4):
        """
        Exact per-pixel multinomial using sequential binomial draws on GPU:
            n1 ~ Binom(M, p1)
            n2 ~ Binom(M-n1, p2/(1-p1))
            n3 ~ Binom(M-n1-n2, p3/(1-p1-p2))
            n4 = residual
        All operations are elementwise on CuPy arrays.
        """
        # Inputs are xp (CuPy) arrays
        # Guards to keep probabilities in [0,1] and avoid /0
        eps = 1e-18
        M_arr = cp.full(p1.shape, M, dtype=cp.int32)

        # First category
        n1 = cp.random.binomial(M_arr, cp.clip(p1, 0.0, 1.0))

        rem1 = M_arr - n1
        p2c = cp.where((1-p1) > eps, cp.clip(p2/(1-p1), 0.0, 1.0), 0.0)
        n2 = cp.random.binomial(cp.asarray(rem1, dtype=cp.int32), p2c)

        rem2 = rem1 - n2
        denom = 1 - p1 - p2
        p3c = cp.where(denom > eps, cp.clip(p3/denom, 0.0, 1.0), 0.0)
        n3 = cp.random.binomial(cp.asarray(rem2, dtype=cp.int32), p3c)

        n4 = rem2 - n3
        return n1, n2, n3, n4

    # ---------------- ICO (Indefinite Causal Order) module ----------------
    # Jones-operator-based quantum switch imaging of commutator contrast C(x,y)
    # and a linear causal-witness functional W(x,y) from the same, single-run dataset.

    @staticmethod
    def _jones_retarder(xp, alpha, delta):
        """
        Linear retarder with fast-axis angle alpha (radians) and retardance delta (radians)
        in H/V basis. Returns 2x2 complex components (each shape (H,W)).
        J = R(α) diag(e^{iΔ/2}, e^{-iΔ/2}) R(-α)
        """
        c = xp.cos(alpha)
        s = xp.sin(alpha)
        e_p = xp.exp(1j*delta/2)
        e_m = xp.exp(-1j*delta/2)
        # Components
        J00 = c*c*e_p + s*s*e_m
        J11 = s*s*e_p + c*c*e_m
        J01 = c*s*(e_p - e_m)
        J10 = J01
        return J00, J01, J10, J11

    @staticmethod
    def _jones_partial_polarizer(xp, tH, tV):
        """
        Partial polarizer diagonal in H/V. tH, tV are intensity transmissions in [0,1].
        Jones matrix uses amplitude transmissions sqrt(t).
        Returns 2x2 complex components (diag real >=0).
        """
        aH = xp.sqrt(xp.clip(tH, 0.0, 1.0))
        aV = xp.sqrt(xp.clip(tV, 0.0, 1.0))
        V00 = aH
        V11 = aV
        V01 = xp.zeros_like(aH, dtype=complex)
        V10 = xp.zeros_like(aH, dtype=complex)
        return V00, V01, V10, V11

    @staticmethod
    def _jones_rotated_partial_polarizer(xp, tH, tV, beta):
        """
        Partial polarizer rotated by beta: V = R(β) diag(√tH, √tV) R(-β)
        Returns 2x2 complex components (each shape (H,W)).
        """
        aH = xp.sqrt(xp.clip(tH, 0.0, 1.0))
        aV = xp.sqrt(xp.clip(tV, 0.0, 1.0))
        cb = xp.cos(beta); sb = xp.sin(beta)
        # R(β) = [[cb, -sb],[sb, cb]], R(-β) = [[cb, sb],[-sb, cb]]
        # Compute V = R diag(aH,aV) R^T(-β)
        # Equivalent elementwise:
        V00 = cb*cb*aH + sb*sb*aV
        V11 = sb*sb*aH + cb*cb*aV
        V01 = cb*sb*(aH - aV)
        V10 = V01
        return V00.astype(complex), V01.astype(complex), V10.astype(complex), V11.astype(complex)

    def _ico_build_UV(self,
                       alpha_mode: str = 'from_phi',
                       alpha_const: float = 0.0,
                       beta_mode: str = 'const',
                       beta_const: float = float(np.pi/8),
                       delta: float = float(np.pi/2),
                       aniso: float = 0.3):
        """
        Construct pixel-wise Jones operators U (retarder) and V (partial polarizer) from current T, φ.
        - alpha_mode: 'from_phi' (default), 'ramp', or 'const'
        - alpha_const: used when mode='const' (radians)
        - delta: retardance (radians) for U
        - aniso: controls V’s anisotropy; tV = clip(T + aniso*(0.8 - T), 0..1)
        Returns tuple of U and V components (U00,U01,U10,U11) and (V00,V01,V10,V11).
        """
        assert self.T is not None and self.phi is not None, "Call load_sample() first."
        xp = self.xp
        H, W = self.H, self.W

        # Alpha field
        if alpha_mode == 'from_phi':
            # Remap φ∈[-π,π) to α∈[0,π) for a rotating fast axis
            alpha = (self.phi + xp.pi) * 0.5
        elif alpha_mode == 'ramp':
            yy, xx = xp.indices((H, W))
            alpha = (xx/(W-1)) * xp.pi
        elif alpha_mode == 'const':
            alpha = xp.full((H,W), float(alpha_const))
        else:
            raise ValueError(f"Unknown alpha_mode: {alpha_mode}")

        # U: linear retarder at alpha with retardance delta
        U00, U01, U10, U11 = self._jones_retarder(xp, alpha, float(delta))

        # V: partial polarizer (rotated by beta). Use T to define tH, and offset tV by 'aniso' towards 0.8
        T = xp.clip(self.T, 0.05, 0.99)
        tH = T
        tV = xp.clip(T + aniso*(0.8 - T), 0.01, 0.99)
        if beta_mode == 'const':
            beta = xp.full((H,W), float(beta_const))
        elif beta_mode == 'from_phi':
            beta = (self.phi + xp.pi) * 0.25  # different mapping than alpha
        elif beta_mode == 'ramp':
            yy, xx = xp.indices((H, W))
            beta = (yy/(H-1)) * xp.pi
        else:
            raise ValueError(f"Unknown beta_mode: {beta_mode}")
        V00, V01, V10, V11 = self._jones_rotated_partial_polarizer(xp, tH, tV, beta)

        return (U00,U01,U10,U11), (V00,V01,V10,V11)

    def _ico_probabilities(self,
                           U, V,
                           vin: Tuple[complex, complex] = (1/np.sqrt(2), 1/np.sqrt(2))):
        """
        Compute per-pixel joint probabilities for outcomes (D1=H,D2=V) × ancilla ±.
        Based on |ψ±⟩ ∝ (UV ± VU)|ψ_in⟩ and subsequent H/V analysis on A.
        Returns four xp arrays p_H_plus, p_V_plus, p_H_minus, p_V_minus that sum to 1 elementwise.
        """
        xp = self.xp
        U00,U01,U10,U11 = U
        V00,V01,V10,V11 = V
        # Since V is diagonal in our construction, speed up products
        # UV = U @ V;  VU = V @ U
        UV00 = U00*V00 + U01*V10  # but V10=0 -> UV00 = U00*V00
        UV01 = U00*V01 + U01*V11  # V01=0     -> UV01 = U01*V11
        UV10 = U10*V00 + U11*V10  # V10=0     -> UV10 = U10*V00
        UV11 = U10*V01 + U11*V11  # V01=0     -> UV11 = U11*V11

        VU00 = V00*U00 + V01*U10  # V01=0 -> VU00 = V00*U00
        VU01 = V00*U01 + V01*U11  # V01=0 -> VU01 = V00*U01
        VU10 = V10*U00 + V11*U10  # V10=0 -> VU10 = V11*U10
        VU11 = V10*U01 + V11*U11  # V10=0 -> VU11 = V11*U11

        # K± = UV ± VU
        Kp00 = UV00 + VU00
        Kp01 = UV01 + VU01
        Kp10 = UV10 + VU10
        Kp11 = UV11 + VU11

        Km00 = UV00 - VU00
        Km01 = UV01 - VU01
        Km10 = UV10 - VU10
        Km11 = UV11 - VU11

        vin0 = complex(vin[0]); vin1 = complex(vin[1])

        # Amplitudes onto H and V for ±
        aH_p = Kp00*vin0 + Kp01*vin1
        aV_p = Kp10*vin0 + Kp11*vin1

        aH_m = Km00*vin0 + Km01*vin1
        aV_m = Km10*vin0 + Km11*vin1

        # Intensities
        I_H_plus  = (aH_p * xp.conj(aH_p)).real
        I_V_plus  = (aV_p * xp.conj(aV_p)).real
        I_H_minus = (aH_m * xp.conj(aH_m)).real
        I_V_minus = (aV_m * xp.conj(aV_m)).real

        # Normalize to probabilities
        tot = I_H_plus + I_V_plus + I_H_minus + I_V_minus
        eps = 1e-18
        tot = xp.where(tot <= 0, eps, tot)
        p_H_plus  = I_H_plus  / tot
        p_V_plus  = I_V_plus  / tot
        p_H_minus = I_H_minus / tot
        p_V_minus = I_V_minus / tot
        return p_H_plus, p_V_plus, p_H_minus, p_V_minus

    def run_ico_acquisition(self,
                             alpha_mode: str = 'from_phi',
                             alpha_const: float = 0.0,
                             beta_mode: str = 'const',
                             beta_const: float = float(np.pi/8),
                             delta: float = float(np.pi/2),
                             aniso: float = 0.3,
                             vin: Tuple[complex, complex] = (1/np.sqrt(2), 1/np.sqrt(2))) -> QEIRunInfo:
        """
        Acquire one ICO run using a quantum-switch model with per-pixel Jones operators U,V.
        Produces joint counts for (D1=H/D2=V) × ancilla outcomes ±.
        Stores counts in key 'ICO'.
        """
        assert self.T is not None and self.phi is not None, "Call load_sample() first."
        M = self.photons_per_pixel
        U, V = self._ico_build_UV(alpha_mode=alpha_mode,
                                  alpha_const=alpha_const,
                                  beta_mode=beta_mode,
                                  beta_const=beta_const,
                                  delta=delta,
                                  aniso=aniso)
        p_H_plus, p_V_plus, p_H_minus, p_V_minus = self._ico_probabilities(U, V, vin)

        if self.use_gpu:
            c1,c2,c3,c4 = self._sample_fourway_gpu(M, p_H_plus, p_V_plus, p_H_minus, p_V_minus)
            self.counts = {'ICO': {
                'D1_plus': asnumpy(c1), 'D2_plus': asnumpy(c2),
                'D1_minus': asnumpy(c3), 'D2_minus': asnumpy(c4)}}
        else:
            c1,c2,c3,c4 = self._sample_fourway_cpu(M, p_H_plus, p_V_plus, p_H_minus, p_V_minus)
            self.counts = {'ICO': {
                'D1_plus': c1, 'D2_plus': c2, 'D1_minus': c3, 'D2_minus': c4}}

        return QEIRunInfo(basis='ICO', theta=None, photons_per_pixel=M)

    # --------- ICO estimators ---------
    @staticmethod
    def reconstruct_commutator_contrast(counts_ICO: Dict[str, np.ndarray], nu: float = 1.0, eps=1e-12):
        """
        Balanced two-port estimator for commutator contrast C(x,y), Eq. (27):
            Ĉ = (ΔR1 - ΔR2) / Σ_j (R(Dj,+)+R(Dj,-)) , optionally divided by visibility ν.
        Also returns the direct ratio estimator C̃ = I−/(I++I−) for reference.
        """
        D1p, D2p = counts_ICO['D1_plus'], counts_ICO['D2_plus']
        D1m, D2m = counts_ICO['D1_minus'], counts_ICO['D2_minus']
        dR1 = D1p - D1m
        dR2 = D2p - D2m
        denom = (D1p + D2p + D1m + D2m) + eps
        C_bal = (dR1 - dR2) / denom
        if nu is not None and nu > 0:
            C_bal = C_bal / float(nu)
        # Direct ratio
        I_minus = D1m + D2m
        I_plus  = D1p + D2p
        C_ratio = I_minus / (I_plus + I_minus + eps)
        return C_bal, C_ratio

    @staticmethod
    def reconstruct_ico_witness(counts_ICO: Dict[str, np.ndarray],
                                weights: Optional[Dict[str, float]] = None,
                                eps=1e-12):
        """
        Evaluate a linear causal-witness functional W = Σ_{o,±} w_{o,±} P(o,±).
        Default weights emphasize the commutator (+/−) asymmetry:
            w(Dj,+) = -1/2, w(Dj,−) = +1/2 for j∈{1,2} → W ∈ [−1, +1].
        Returns W on [−1, +1] approximately.
        """
        D1p, D2p = counts_ICO['D1_plus'], counts_ICO['D2_plus']
        D1m, D2m = counts_ICO['D1_minus'], counts_ICO['D2_minus']
        tot = (D1p + D2p + D1m + D2m) + eps
        P = {
            'D1_plus':  D1p / tot,
            'D2_plus':  D2p / tot,
            'D1_minus': D1m / tot,
            'D2_minus': D2m / tot,
        }
        if weights is None:
            weights = {'D1_plus': -0.5, 'D2_plus': -0.5, 'D1_minus': +0.5, 'D2_minus': +0.5}
        W = sum(weights[k]*P[k] for k in P.keys())
        return W

    # --------- acquisition ---------
    def run_acquisition(self, analyzer_angle: Optional[float] = None) -> QEIRunInfo:
        """
        Acquire one run at a given analyzer setting:
            None: H/V basis
            pi/4: D/A basis
            else: rotated orthonormal basis at angle theta=analyzer_angle
        """
        assert self.T is not None and self.phi is not None, "Call load_sample() first."
        M = self.photons_per_pixel

        if analyzer_angle is None:
            basis, theta = 'HV', None
            a,b,c,d = self._intensities_HV(self.T)
            p1,p2,p3,p4 = self._normalize_four(a,b,c,d)
            if self.use_gpu:
                c1,c2,c3,c4 = self._sample_fourway_gpu(M, p1, p2, p3, p4)
                # Move counts to CPU for plotting/analysis
                self.counts = {'HV': {
                    'D1_H': asnumpy(c1), 'D2_H': asnumpy(c2),
                    'D1_V': asnumpy(c3), 'D2_V': asnumpy(c4)}}
            else:
                c1,c2,c3,c4 = self._sample_fourway_cpu(M, p1, p2, p3, p4)
                self.counts = {'HV': {
                    'D1_H': c1, 'D2_H': c2, 'D1_V': c3, 'D2_V': c4}}

        elif abs(analyzer_angle - np.pi/4) < 1e-12:
            basis, theta = 'DA', None
            a,b,c,d = self._intensities_DA(self.T, self.phi)
            p1,p2,p3,p4 = self._normalize_four(a,b,c,d)
            if self.use_gpu:
                c1,c2,c3,c4 = self._sample_fourway_gpu(M, p1, p2, p3, p4)
                self.counts = {'DA': {
                    'D1_D': asnumpy(c1), 'D2_D': asnumpy(c2),
                    'D1_A': asnumpy(c3), 'D2_A': asnumpy(c4)}}
            else:
                c1,c2,c3,c4 = self._sample_fourway_cpu(M, p1, p2, p3, p4)
                self.counts = {'DA': {
                    'D1_D': c1, 'D2_D': c2, 'D1_A': c3, 'D2_A': c4}}

        else:
            basis, theta = 'ROT', float(analyzer_angle)
            a,b,c,d = self._intensities_ROT(self.T, self.phi, theta)
            p1,p2,p3,p4 = self._normalize_four(a,b,c,d)
            if self.use_gpu:
                c1,c2,c3,c4 = self._sample_fourway_gpu(M, p1, p2, p3, p4)
                self.counts = {f'ROT:{theta:.6f}': {
                    'D1_e1': asnumpy(c1), 'D2_e1': asnumpy(c2),
                    'D1_e2': asnumpy(c3), 'D2_e2': asnumpy(c4)}}
            else:
                c1,c2,c3,c4 = self._sample_fourway_cpu(M, p1, p2, p3, p4)
                self.counts = {f'ROT:{theta:.6f}': {
                    'D1_e1': c1, 'D2_e1': c2, 'D1_e2': c3, 'D2_e2': c4}}

        return QEIRunInfo(basis=basis, theta=theta, photons_per_pixel=M)

    # --------- estimators ---------
    @staticmethod
    def reconstruct_absorption(counts_HV: Dict[str, np.ndarray], eps=1e-12):
        nH = counts_HV['D1_H'] + counts_HV['D2_H']
        nV = counts_HV['D1_V'] + counts_HV['D2_V']
        T_hat = (nH + eps) / (nV + eps)
        sigma_T = T_hat * np.sqrt(np.maximum(0.0, 1.0/(nH+eps) + 1.0/(nV+eps)))
        return T_hat, sigma_T

    @staticmethod
    def reconstruct_visibility_DA(counts_DA: Dict[str, np.ndarray], eps=1e-12):
        D1D, D2D = counts_DA['D1_D'], counts_DA['D2_D']
        D1A, D2A = counts_DA['D1_A'], counts_DA['D2_A']
        num = (D1D - D1A) - (D2D - D2A)
        den = (D1D + D1A) + (D2D + D2A)
        V = np.where(den > 0, num/(den+eps), 0.0)
        N = den
        sigma_V = np.sqrt(np.maximum(0.0, (1.0 - np.clip(V, -1, 1)**2) / (N + eps)))
        return V, sigma_V

    @staticmethod
    def reconstruct_visibility_ROT(counts_ROT: Dict[str, np.ndarray], eps=1e-12):
        D1e1, D2e1 = counts_ROT['D1_e1'], counts_ROT['D2_e1']
        D1e2, D2e2 = counts_ROT['D1_e2'], counts_ROT['D2_e2']
        num = (D1e1 - D1e2) - (D2e1 - D2e2)
        den = (D1e1 + D1e2) + (D2e1 + D2e2)
        V = np.where(den > 0, num/(den+eps), 0.0)
        N = den
        sigma_V = np.sqrt(np.maximum(0.0, (1.0 - np.clip(V, -1, 1)**2) / (N + eps)))
        return V, sigma_V

    @staticmethod
    def amplitude_A_theta(T_hat: np.ndarray, theta: float, eps=1e-12) -> np.ndarray:
        T_safe = np.clip(T_hat, eps, None)
        return (2.0*np.sqrt(T_safe)*np.sin(2*theta)) / (1.0 + T_safe)

    @staticmethod
    def phase_from_visibility(V: np.ndarray, T_hat: np.ndarray, theta: float, eps=1e-12) -> np.ndarray:
        A = QEISimulator.amplitude_A_theta(T_hat, theta, eps=eps)
        cosphi = np.zeros_like(V)
        m = A > 1e-9
        cosphi[m] = np.clip(V[m]/A[m], -1.0, 1.0)
        return np.arccos(cosphi)  # modulo pi

    @staticmethod
    def phase_from_two_steps(V0: np.ndarray, Vd: np.ndarray, delta: float, eps=1e-12) -> np.ndarray:
        """
        Two-step retrieval (amplitude cancels):
        tan φ = (V0 sin δ) / (V0 cos δ - Vδ)
        Returns φ in [0, 2π). You can remap to [-π, π) if desired.
        """
        num = V0*np.sin(delta)
        den = V0*np.cos(delta) - Vd
        den_safe = np.where(np.abs(den) < eps, eps, den)
        den_safe = np.copysign(np.abs(den_safe), den)
        phi = np.arctan2(num, den_safe)
        return (phi + 2*np.pi) % (2*np.pi)

    # --------- sanity checks ---------
    @staticmethod
    def check_port_sum_invariance(counts: Dict[str, np.ndarray]) -> float:
        """
        Returns the mean absolute difference between (D1 sum) and (D2 sum)
        across pixels for the provided basis counts.
        Expect ~0 (up to shot noise) if formulas are correct.
        """
        keys = list(counts.keys())
        D1sum = None; D2sum = None
        for k in keys:
            if k.startswith('D1_'):
                D1sum = counts[k] if D1sum is None else D1sum + counts[k]
            elif k.startswith('D2_'):
                D2sum = counts[k] if D2sum is None else D2sum + counts[k]
        if D1sum is None or D2sum is None:
            return float('nan')
        return float(np.mean(np.abs(D1sum - D2sum)))

# ---------------- plotting helper ----------------
def create_publication_figure(sim: QEISimulator,
                              sample_image: Optional[str] = None,
                              theta: float = np.pi/8,
                              phi_mode: str = "smooth",
                              phi_image: Optional[str] = None,
                              T_const: Optional[float] = None,
                              phi_step_thresh: float = 0.5,
                              delta: Optional[float] = None,
                              # ICO options
                              ico: bool = False,
                              ico_alpha_mode: str = 'from_phi',
                              ico_alpha_const: float = 0.0,
                              ico_beta_mode: str = 'const',
                              ico_beta_const: float = float(np.pi/8),
                              ico_delta: float = float(np.pi/2),
                              ico_aniso: float = 0.3,
                              ico_nu: float = 1.0,
                              ico_witness_weights: Optional[Dict[str, float]] = None,
                              ico_vin_psi: float = float(np.pi/4),
                              ico_vin_xi: float = 0.0):
    sim.load_sample(sample_image, phi_mode=phi_mode, phi_image=phi_image,
                    T_const=T_const, phi_step_thresh=phi_step_thresh)

    # Acquisition runs
    sim.run_acquisition(analyzer_angle=None)           # HV
    T_hat, T_err = QEISimulator.reconstruct_absorption(sim.counts['HV'])

    sim.run_acquisition(analyzer_angle=np.pi/4)        # D/A
    V_DA, V_DA_err = QEISimulator.reconstruct_visibility_DA(sim.counts['DA'])
    hv_diff = QEISimulator.check_port_sum_invariance(sim.counts['DA'])

    theta_is_da = np.isclose(theta, np.pi/4, atol=1e-9)

    if theta_is_da:
        # Reuse the D/A measurement when θ coincides with π/4 to avoid key mismatches.
        V_th, V_th_err = V_DA, V_DA_err
    else:
        sim.run_acquisition(analyzer_angle=theta)          # Rotated
        key = list(sim.counts.keys())[0]
        V_th, V_th_err = QEISimulator.reconstruct_visibility_ROT(sim.counts[key])

    phi_hat = QEISimulator.phase_from_visibility(V_th, T_hat, theta)

    # Two-step retrieval (optional)
    phi_two = None
    if delta is not None:
        # Save current phi, add global step, acquire again at same θ
        phi_saved = sim.phi.copy()
        sim.phi = sim.phi + delta
        if theta_is_da:
            sim.run_acquisition(analyzer_angle=np.pi/4)
            V_th_delta, _ = QEISimulator.reconstruct_visibility_DA(sim.counts['DA'])
        else:
            sim.run_acquisition(analyzer_angle=theta)
            key2 = list(sim.counts.keys())[0]
            V_th_delta, _ = QEISimulator.reconstruct_visibility_ROT(sim.counts[key2])
        phi_two = QEISimulator.phase_from_two_steps(V_th, V_th_delta, delta)
        # restore
        sim.phi = phi_saved

    # Build figure
    T_true = asnumpy(sim.T); phi_true = asnumpy(sim.phi)
    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(T_true, vmin=0, vmax=1, cmap='gray'); ax1.set_title('True Transmission T'); ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(phi_true, vmin=-np.pi, vmax=np.pi, cmap='twilight'); ax2.set_title('True Phase φ'); ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = plt.subplot(3, 4, 5)
    im3 = ax3.imshow(T_hat, vmin=0, vmax=1, cmap='gray'); ax3.set_title('Reconstructed T (H/V)'); ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    ax4 = plt.subplot(3, 4, 6)
    im4 = ax4.imshow(T_err, cmap='Reds'); ax4.set_title('Uncertainty σ_T'); ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    ax5 = plt.subplot(3, 4, 9)
    im5 = ax5.imshow(V_DA, vmin=-1, vmax=1, cmap='RdBu_r'); ax5.set_title('Visibility V (D/A)'); ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    ax6 = plt.subplot(3, 4, 10)
    im6 = ax6.imshow(V_DA_err, cmap='Reds'); ax6.set_title('Uncertainty σ_V (D/A)'); ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    # Distributions & correlation
    ax7 = plt.subplot(3, 4, 3)
    ax7.hist(T_hat.ravel(), bins=30, alpha=0.6, label='Reconstructed', density=True)
    ax7.hist(T_true.ravel(), bins=30, alpha=0.6, label='True', density=True)
    ax7.set_title('T Distribution'); ax7.legend(); ax7.grid(alpha=0.3)

    ax8 = plt.subplot(3, 4, 4)
    ax8.scatter(T_true.ravel(), T_hat.ravel(), s=2, alpha=0.3)
    ax8.plot([0,1],[0,1],'--',alpha=0.5)
    ax8.set_xlabel('True T'); ax8.set_ylabel('Reconstructed T')
    ax8.set_title(f'Correlation r={np.corrcoef(T_true.ravel(), T_hat.ravel())[0,1]:.3f}')
    ax8.grid(alpha=0.3)

    # Rotated-basis outputs
    ax11 = plt.subplot(3, 4, 11)
    im11 = ax11.imshow(V_th, vmin=-0.5, vmax=0.5, cmap='RdBu_r'); ax11.set_title(f'Vθ at θ={theta:.3f}'); ax11.axis('off')
    plt.colorbar(im11, ax=ax11, fraction=0.046)

    ax12 = plt.subplot(3, 4, 12)
    im12 = ax12.imshow(phi_hat, vmin=0, vmax=np.pi, cmap='twilight')
    ax12.set_title('φ̂ from Vθ (mod π)'); ax12.axis('off')
    plt.colorbar(im12, ax=ax12, fraction=0.046)

    # Fisher Information vs θ (Eq. 19; representative T, φ=π/2 gives X=0)
    ax10 = plt.subplot(3, 4, 8)
    angles = np.linspace(0, np.pi/2, 50)
    T_typ = float(np.median(T_true))
    phi_typ = np.pi/2
    Ivals = []
    for th in angles:
        c, s = np.cos(th), np.sin(th)
        A = T_typ*c*c + s*s
        B = T_typ*s*s + c*c
        X = 2*np.sqrt(T_typ)*s*c*np.cos(phi_typ)
        I = (4*T_typ*s*s*c*c*np.sin(phi_typ)**2)/(T_typ+1) * (A/(A*A - X*X) + B/(B*B - X*X))
        Ivals.append(I)
    ax10.plot(angles*180/np.pi, Ivals, 'o-')
    ax10.set_xlabel('θ (deg)'); ax10.set_ylabel('I_φ(θ)'); ax10.set_title('Fisher Information vs θ')
    ax10.grid(alpha=0.3)

    plt.suptitle(f'Quantum Erasure 2 Photon Imaging — Monte Carlo of: HV, D/A, Rotated basis |  port-sum Δ≈{hv_diff:.3g}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Optional second figure for two-step φ
    fig2 = None
    if phi_two is not None:
        fig2 = plt.figure(figsize=(10,4))
        ax21 = plt.subplot(1,2,1)
        im21 = ax21.imshow(phi_two, vmin=0, vmax=2*np.pi, cmap='twilight')
        ax21.set_title(f'φ̂ (two-step)  δ={delta:.3f}'); ax21.axis('off')
        plt.colorbar(im21, ax=ax21, fraction=0.046)
        ax22 = plt.subplot(1,2,2)
        # Compare against true φ modulo 2π (shift negative to [0,2π))
        phi_true_2pi = (phi_true + 2*np.pi) % (2*np.pi)
        im22 = ax22.imshow(phi_true_2pi, vmin=0, vmax=2*np.pi, cmap='twilight')
        ax22.set_title('True φ (mod 2π)'); ax22.axis('off')
        plt.colorbar(im22, ax=ax22, fraction=0.046)
        plt.tight_layout()

    # Optional ICO figure
    fig_ico = None
    if ico:
        # Run ICO acquisition and reconstruct C and W
        # Input polarization |ψ_in> = [cosψ, e^{iξ} sinψ]
        vin = (np.cos(ico_vin_psi), np.exp(1j*ico_vin_xi)*np.sin(ico_vin_psi))
        sim.run_ico_acquisition(alpha_mode=ico_alpha_mode,
                                alpha_const=ico_alpha_const,
                                beta_mode=ico_beta_mode,
                                beta_const=ico_beta_const,
                                delta=ico_delta,
                                aniso=ico_aniso,
                                vin=vin)
        C_bal, C_ratio = QEISimulator.reconstruct_commutator_contrast(sim.counts['ICO'], nu=ico_nu)
        W_map = QEISimulator.reconstruct_ico_witness(sim.counts['ICO'], weights=ico_witness_weights)

        C_bal = asnumpy(C_bal)
        C_ratio = asnumpy(C_ratio)
        W_map = asnumpy(W_map)

        fig_ico = plt.figure(figsize=(12,4))
        axI1 = plt.subplot(1,3,1)
        imI1 = axI1.imshow(C_bal, vmin=0, vmax=1, cmap='magma')
        axI1.set_title('ICO: Commutator Contrast Ĉ (balanced)'); axI1.axis('off')
        plt.colorbar(imI1, ax=axI1, fraction=0.046)

        axI2 = plt.subplot(1,3,2)
        imI2 = axI2.imshow(C_ratio, vmin=0, vmax=1, cmap='magma')
        axI2.set_title('ICO: C̃ = I−/(I++I−)'); axI2.axis('off')
        plt.colorbar(imI2, ax=axI2, fraction=0.046)

        axI3 = plt.subplot(1,3,3)
        imI3 = axI3.imshow(W_map, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        axI3.set_title('ICO: Causal Witness W'); axI3.axis('off')
        plt.colorbar(imI3, ax=axI3, fraction=0.046)
        plt.tight_layout()

    return (fig, fig2 if phi_two is not None else None, fig_ico)

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Quantum Erasure Imaging + ICO (Quantum-Switch) - Monte Carlo (GPU-accelerated)")
    parser.add_argument('--H', type=int, default=256, help='Image height')
    parser.add_argument('--W', type=int, default=256, help='Image width')
    parser.add_argument('--ppp', type=int, default=100, help='Photons per pixel')
    parser.add_argument('--theta', type=float, default=float(np.pi/8), help='Rotated analyzer angle (radians)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU via CuPy if available')

    # amplitude T and phi options
    parser.add_argument('--image', type=str, default=None, help='Optional path to grayscale image for T')
    parser.add_argument('--T-const', type=float, default=None, help='Override T with a constant in [0.1,0.99]')
    parser.add_argument('--phi-mode', type=str, default='smooth',
                        choices=['smooth','ramp','vortex','zernike','hue','gray','step'],
                        help='How to generate the phase map φ')
    parser.add_argument('--phi-image', type=str, default=None, help='Optional image to drive phase (hue/gray/step modes)')
    parser.add_argument('--phi-step-thresh', type=float, default=0.5, help='Threshold in [0,1] for step phase from grayscale')

    # two-step retrieval
    parser.add_argument('--delta', type=float, default=None, help='Global phase step δ (radians) for two-step φ retrieval')

    # ICO (quantum switch) options
    parser.add_argument('--ico', action='store_true', help='Run ICO module (quantum switch) and plot commutator contrast and witness')
    parser.add_argument('--ico-alpha-mode', type=str, default='from_phi', choices=['from_phi','ramp','const'], help='Fast-axis field α(x,y) for U')
    parser.add_argument('--ico-alpha-const', type=float, default=0.0, help='Constant α when alpha-mode=const (radians)')
    parser.add_argument('--ico-beta-mode', type=str, default='const', choices=['from_phi','ramp','const'], help='Axis field β(x,y) for V')
    parser.add_argument('--ico-beta-const', type=float, default=float(np.pi/8), help='Constant β when beta-mode=const (radians)')
    parser.add_argument('--ico-delta', type=float, default=float(np.pi/2), help='Retardance Δ of U (radians)')
    parser.add_argument('--ico-aniso', type=float, default=0.3, help='Anisotropy of V (tV shift toward 0.8)')
    parser.add_argument('--ico-nu', type=float, default=1.0, help='Experimental visibility factor ν for Ĉ balancing')
    # Witness weights as four floats if desired
    parser.add_argument('--ico-witness-weights', type=float, nargs=4, metavar=('wD1+','wD2+','wD1-','wD2-'), default=None,
                        help='Optional witness weights for (D1+, D2+, D1-, D2-)')
    parser.add_argument('--ico-vin-psi', type=float, default=float(np.pi/4), help='Input polarization |ψ_in> orientation ψ (radians)')
    parser.add_argument('--ico-vin-xi', type=float, default=0.0, help='Input polarization phase ξ between H/V (radians)')

    parser.add_argument('--nofig', action='store_true', help='Do not display the figure')
    parser.add_argument('--save', action='store_true', help='Save figures to files instead of (or in addition to) showing')
    parser.add_argument('--save-prefix', type=str, default='qei', help='Filename prefix for saved figures')
    args = parser.parse_args()

    use_gpu = bool(args.gpu and _CUPY_AVAILABLE)
    if args.gpu and not _CUPY_AVAILABLE:
        print("[warning] --gpu specified but CuPy not available; running on CPU.]")

    sim = QEISimulator(image_size=(args.H, args.W),
                       photons_per_pixel=args.ppp,
                       use_gpu=use_gpu)

    # Pack witness weights if provided
    witness_weights = None
    if args.ico_witness_weights is not None:
        w = args.ico_witness_weights
        witness_weights = {'D1_plus': w[0], 'D2_plus': w[1], 'D1_minus': w[2], 'D2_minus': w[3]}

    fig, fig2, fig_ico = create_publication_figure(sim,
                                  sample_image=args.image,
                                  theta=args.theta,
                                  phi_mode=args.phi_mode,
                                  phi_image=args.phi_image,
                                  T_const=args.T_const,
                                  phi_step_thresh=args.phi_step_thresh,
                                  delta=args.delta,
                                  ico=args.ico,
                                  ico_alpha_mode=args.ico_alpha_mode,
                                  ico_alpha_const=args.ico_alpha_const,
                                  ico_beta_mode=args.ico_beta_mode,
                                  ico_beta_const=args.ico_beta_const,
                                  ico_delta=args.ico_delta,
                                  ico_aniso=args.ico_aniso,
                                  ico_nu=args.ico_nu,
                                  ico_witness_weights=witness_weights,
                                  ico_vin_psi=args.ico_vin_psi,
                                  ico_vin_xi=args.ico_vin_xi)

    # Save figures if requested
    if args.save:
        prefix = args.save_prefix
        try:
            fig.savefig(f"{prefix}_main.png", dpi=150)
        except Exception as e:
            print(f"[warn] Could not save {prefix}_main.png: {e}")
        if fig2 is not None:
            try:
                fig2.savefig(f"{prefix}_twostep.png", dpi=150)
            except Exception as e:
                print(f"[warn] Could not save {prefix}_twostep.png: {e}")
        if fig_ico is not None:
            try:
                fig_ico.savefig(f"{prefix}_ico.png", dpi=150)
            except Exception as e:
                print(f"[warn] Could not save {prefix}_ico.png: {e}")

    if not args.nofig:
        plt.show()

if __name__ == "__main__":
    main()

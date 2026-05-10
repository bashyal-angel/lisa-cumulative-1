# lisa-cumulative-1
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import os

# ========================= PARAMETERS =========================
T_days = 90.0
fs = 2.0
dt = 1.0 / fs
T_sec = T_days * 86400.0
t = np.arange(0, T_sec, dt)
N = len(t)
print(f"Running {T_days} days simulation, {N:,} samples")

L = 8.33                  # one-way light time [s]
mu = 0.3                  # \hat{k} · n
delay_gw = (1 - mu) * L

# GW source
f_gw = 3e-3               # Hz
h_plus = 1e-21            # amplitude for +
h_cross = 0.5e-21         # amplitude for × (can be different)

# ========================= POLARIZATION TENSORS & PATTERN =========================
def compute_pattern(mu, psi=0.0):
    """Compute F+ and Fx for given mu = k·n and polarization angle psi"""
    # Simple analytic form for arm response (standard approximation)
    # Full version would use arm vector n and sky direction k
    cos2psi = np.cos(2*psi)
    sin2psi = np.sin(2*psi)
    F_plus  = 0.5 * (1 + mu**2) * cos2psi   # rough, but illustrative
    F_cross = mu * sin2psi
    return F_plus, F_cross

F_plus, F_cross = compute_pattern(mu)

print(f"Antenna patterns: F+ = {F_plus:.3f}, Fx = {F_cross:.3f}")

# ========================= GW STRAIN (Full tensor projection) =========================
class GWSource:
    def __init__(self, f_gw, h_plus, h_cross, phi=0.0):
        self.f_gw = f_gw
        self.h_plus = h_plus
        self.h_cross = h_cross
        self.phi = phi
    
    def strain(self, tau, pol='plus'):
        arg = 2 * np.pi * self.f_gw * tau + self.phi
        if pol == 'plus':
            return self.h_plus * np.sin(arg)
        else:
            return self.h_cross * np.sin(arg)
    
    def one_way_gw(self, tau_r, L, mu=mu):
        """Exact match to PDF Eq. (9) structure"""
        dt = (1 - mu) * L
        h_ab_r   = F_plus * self.strain(tau_r, 'plus') + F_cross * self.strain(tau_r, 'cross')
        h_ab_em  = F_plus * self.strain(tau_r - dt, 'plus') + F_cross * self.strain(tau_r - dt, 'cross')
        diff = h_ab_r - h_ab_em
        return diff / (2 * (1 - mu))   # as in your PDF Eq. (9)

gw = GWSource(f_gw, h_plus, h_cross)

# ========================= NOISE GENERATION =========================
np.random.seed(42)

def generate_colored_noise(t, rms, alpha=-2, highpass=False, fs=fs):
    N = len(t)
    freqs = np.fft.rfftfreq(N, 1/fs)
    freqs[0] = freqs[1]
    psd = (freqs ** alpha)
    psd[0] = psd[1]
    psd /= np.max(psd)
    psd *= (rms ** 2 * N / fs)
    white = np.random.randn(N)
    fft_white = np.fft.rfft(white)
    fft_colored = fft_white * np.sqrt(psd)
    noise = np.real(np.fft.irfft(fft_colored, n=N))
    if highpass:
        from scipy.signal import butter, lfilter
        b, a = butter(2, 0.01, btype='high', fs=fs)
        noise = lfilter(b, a, noise)
    return noise

laser_rms = 1e-12
secondary_rms = 1e-15
servo_rms = 5e-15

C1 = generate_colored_noise(t, laser_rms)
C2 = generate_colored_noise(t, laser_rms)
eta12 = generate_colored_noise(t, secondary_rms, alpha=-1)
Nservo = generate_colored_noise(t, servo_rms, alpha=-1, highpass=True)

# ========================= ONE-WAY MEASUREMENTS =========================
ygw_12 = gw.one_way_gw(t, L)
ygw_21 = gw.one_way_gw(t, L)   # symmetric for this toy

def measured_oneway(ygw, Cs, Cr, eta, delay=L):
    return ygw + np.interp(t - delay, t, Cs) - Cr + eta

y12 = measured_oneway(ygw_12, C1, C2, eta12)
y21 = measured_oneway(ygw_21, C2, C1, eta12)

# ========================= RETURNED ECHO =========================
y_returned = (ygw_21 + 
              np.interp(t - L, t, ygw_12) +
              np.interp(t - 2*L, t, C1) - C1 +
              np.interp(t - L, t, Nservo) + eta12)

# Free-running reference (no explicit servo/phase-lock)
y_free = y21 + np.interp(t - L, t, y12) + np.interp(t - 2*L, t, C1) - C1 + eta12

# ========================= TDI =========================
delay2 = 2 * L
X_tdi_returned = y_returned - np.interp(t - delay2, t, y_returned)
X_tdi_free     = y_free     - np.interp(t - delay2, t, y_free)

# ========================= SNR =========================
def get_psd(data, fs=fs, nperseg=2**16):
    f, Pxx = welch(data, fs=fs, nperseg=nperseg, window='hann', average='median')
    return f, Pxx

f_psd, psd_res = get_psd(X_tdi_returned[:int(0.8*N)])   # use part of data for PSD

def compute_snr(gw_signal, psd_f, psd_val, fs=fs):
    freqs = np.fft.rfftfreq(len(gw_signal), 1/fs)
    df = freqs[1]-freqs[0]
    fft_sig = np.fft.rfft(gw_signal)
    psd_interp = np.interp(freqs, psd_f, psd_val, left=psd_val[0], right=psd_val[-1])
    snr2 = 4.0 * np.sum(np.abs(fft_sig)**2 / (psd_interp + 1e-30)) * df
    return np.sqrt(snr2)

gw_tdi_approx = gw.one_way_gw(t, L)   # approximate TDI response
snr_returned = compute_snr(gw_tdi_approx, f_psd, psd_res)
snr_free = compute_snr(gw_tdi_approx, *get_psd(X_tdi_free[:int(0.8*N)]))

print(f"SNR (Phase-locked with servo, Eq.12): {snr_returned:.2f}")
print(f"SNR (Free-running reference): {snr_free:.2f}")

# ========================= PLOTTING =========================
plt.figure(figsize=(14, 10))
plt.subplot(3,1,1)
plt.plot(t[:20000], y_returned[:20000])
plt.title('Raw Returned Measurement (Your Eq. 12)')
plt.subplot(3,1,2)
plt.plot(t[:20000], X_tdi_returned[:20000], label='TDI with Phase-Lock')
plt.plot(t[:20000], X_tdi_free[:20000], alpha=0.6, label='TDI Free-running')
plt.legend()
plt.title('Post-TDI X Channel')
plt.subplot(3,1,3)
plt.loglog(f_psd[1:], np.sqrt(psd_res[1:]), label='Residual ASD')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude Spectral Density')
plt.legend()
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/lisa_eq12_tdi.png", dpi=300)
print("Plot saved to results/lisa_eq12_tdi.png")

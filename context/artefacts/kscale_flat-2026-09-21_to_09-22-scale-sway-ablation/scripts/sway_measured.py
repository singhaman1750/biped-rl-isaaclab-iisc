"""The lateral torso excursion the three policies already produce, from the base frame velocity.

base_linear_velocity is recorded in the BASE frame, so its second column is the lateral
component directly.  Subtracting the commanded lateral velocity leaves the oscillation, whose
amplitude at the stride frequency converts to a displacement amplitude by dividing by 2*pi*f.
"""
import numpy as np, common
for run in common.RUNS:
    d = common.load(run)
    dt = float(d["step_dt"])
    vy  = d["base_linear_velocity"][:,:,1]
    cy  = d["commanded_linear_velocity"][:,:,1]
    f_gait = d["gait_command"][:,:,0]
    res = vy - cy
    T, E = res.shape
    fs = 1.0/dt
    freqs = np.fft.rfftfreq(T, dt)
    amps = []
    for e in range(E):
        x = res[:,e] - res[:,e].mean()
        S = np.abs(np.fft.rfft(x)) * 2.0 / T          # single sided velocity amplitude
        f0 = float(np.median(f_gait[:,e]))            # stride frequency for this env
        band = (freqs > 0.6*f0) & (freqs < 1.6*f0)
        k = np.argmax(S[band]); fpk = freqs[band][k]; vamp = S[band][k]
        amps.append((fpk, vamp, vamp/(2*np.pi*fpk)))
    fpk, vamp, damp = np.mean(amps, axis=0)
    print(f"{run:9} gait f {np.median(f_gait):.3f} Hz | peak at {fpk:.3f} Hz | lateral velocity amplitude "
          f"{vamp:.4f} m/s | displacement amplitude {damp:.4f} m")
    print(f"{'':9} residual lateral speed RMS {res.std():.4f} m/s, total lateral excursion needed 0.026 to 0.037 m")

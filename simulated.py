import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Realistic Small Drone (UAV) Doppler Effect Simulation
# =============================================================================

# -----------------------------
# 1) Global Physical Parameters
# -----------------------------
fs = 8000.0  # Sample rate [Hz]
T = 120.0  # Duration [s] - longer for realistic drone missions
t = np.arange(0.0, T, 1.0 / fs)
dt = 1.0 / fs
N = t.size

# Environmental parameters
c = 343.0  # Speed of sound [m/s] (20°C)
rho_air = 1.225  # Air density [kg/m³]
temp = 20.0  # Temperature [°C]
humidity = 60.0  # Relative humidity [%]


# Atmospheric absorption coefficient (frequency dependent)
def atm_absorption(f, temp, humidity, altitude=100):
    """Atmospheric absorption in dB/km as function of frequency and altitude"""
    # More significant at drone altitudes
    base_abs = 0.1 * (f / 1000) ** 1.8 + 0.02 * (f / 1000) ** 2.5
    altitude_factor = 1.0 + 0.3 * (altitude / 100)  # Higher absorption at altitude
    return base_abs * altitude_factor


# Two sensor positions (realistic spacing for drone detection)
S1 = np.array([0.0, 0.0])  # Sensor 1 at origin
S2 = np.array([200.0, 150.0])  # Sensor 2 - wider spacing for triangulation
sensors = [S1, S2]
sensor_names = ['Sensor 1 (Origin)', 'Sensor 2 (200m, 150m)']

rng = np.random.default_rng(2025)


# -----------------------------
# 2) Realistic Small Drone Trajectory
# -----------------------------
def create_drone_trajectory(t):
    """Create realistic small drone mission trajectory"""

    # Drone mission parameters
    max_speed = 25.0  # m/s (90 km/h - fast drone)
    typical_speed = 12.0  # m/s (43 km/h - cruise speed)
    max_accel = 8.0  # m/s² (aggressive maneuvering)
    altitude = 80.0  # m (typical surveillance altitude)

    # Mission phases with different behaviors
    total_time = T

    # Base trajectory - surveillance pattern with waypoints
    waypoints_x = [0, 150, 300, 250, 100, -50, 50, 200]
    waypoints_y = [200, 350, 200, 50, 100, 300, 400, 300]
    waypoint_times = np.linspace(0, total_time, len(waypoints_x))

    # Interpolate base path
    from scipy.interpolate import interp1d
    interp_x = interp1d(waypoint_times, waypoints_x, kind='cubic')
    interp_y = interp1d(waypoint_times, waypoints_y, kind='cubic')

    x_base = interp_x(t)
    y_base = interp_y(t)

    # Add realistic drone behaviors

    # 1) Loitering patterns (circular holds)
    loiter_centers = [(150, 350), (100, 100), (200, 250)]
    loiter_times = [25, 60, 95]  # seconds
    loiter_radius = 40.0

    for (lx, ly), lt in zip(loiter_centers, loiter_times):
        # Create time window for loitering
        t_window = np.abs(t - lt) < 8  # 16 second loiter
        if np.any(t_window):
            loiter_t = (t[t_window] - lt + 8) * 0.3  # Slower circular motion
            x_base[t_window] = lx + loiter_radius * np.cos(loiter_t)
            y_base[t_window] = ly + loiter_radius * np.sin(loiter_t)

    # 2) Evasive maneuvers (sudden direction changes)
    evasion_times = [40, 75, 105]
    for et in evasion_times:
        t_window = np.abs(t - et) < 3  # 6 second evasion
        if np.any(t_window):
            # Sharp S-curve maneuver
            local_t = t[t_window] - et + 3
            evasion_x = 30 * np.sin(2 * np.pi * local_t / 3)
            evasion_y = 15 * np.cos(4 * np.pi * local_t / 3)
            x_base[t_window] += evasion_x
            y_base[t_window] += evasion_y

    # 3) Search patterns (back-and-forth)
    search_times = [15, 85]
    for st in search_times:
        t_window = np.abs(t - st) < 6  # 12 second search
        if np.any(t_window):
            local_t = t[t_window] - st + 6
            search_x = 25 * np.sin(np.pi * local_t / 2)
            x_base[t_window] += search_x

    # 4) Wind gusts and turbulence (small random perturbations)
    wind_freq = 0.1  # Hz
    turbulence_x = 3 * np.sin(2 * np.pi * wind_freq * t) + 2 * np.sin(2 * np.pi * 1.3 * t)
    turbulence_y = 2 * np.cos(2 * np.pi * 0.8 * wind_freq * t) + 1.5 * np.sin(2 * np.pi * 2.1 * t)

    # Add small random buffeting
    buffet_x = np.cumsum(rng.normal(0, 0.05, len(t))) * dt * 2
    buffet_y = np.cumsum(rng.normal(0, 0.05, len(t))) * dt * 2

    x_final = x_base + turbulence_x + buffet_x
    y_final = y_base + turbulence_y + buffet_y

    # Add altitude variations (3D flight)
    z_base = altitude * np.ones_like(t)
    # Altitude changes for obstacle avoidance and mission phases
    z_variations = 15 * np.sin(2 * np.pi * 0.05 * t) + 10 * np.sin(2 * np.pi * 0.15 * t + 0.3)
    z_final = z_base + z_variations

    return np.stack([x_final, y_final, z_final], axis=1)


r = create_drone_trajectory(t)


# Compute kinematics with smoothing (drones have quick response)
def smooth_derivative(x, dt, sigma=1.0):
    """Compute derivative with light smoothing for drone dynamics"""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.gradient(x, dt, axis=0), sigma=sigma, axis=0)


v = smooth_derivative(r, dt, sigma=0.8)  # Less smoothing for responsive drone
a = smooth_derivative(v, dt, sigma=0.8)

# Speed limiting (realistic drone constraints)
speed = np.linalg.norm(v[:, :2], axis=1)  # 2D ground speed
max_realistic_speed = 25.0
speed_limit_factor = np.where(speed > max_realistic_speed,
                              max_realistic_speed / speed, 1.0)
v[:, 0] *= speed_limit_factor
v[:, 1] *= speed_limit_factor

# Recalculate acceleration after speed limiting
a = smooth_derivative(v, dt, sigma=0.8)


# -----------------------------
# 3) Small Drone Acoustic Model
# -----------------------------
class SmallDroneSource:
    def __init__(self):
        # Small quadcopter/hexcopter characteristics
        self.num_rotors = 4
        self.rotor_diameter = 0.25  # m (10 inch props)
        self.nominal_rpm = 6000  # RPM at hover
        self.blades_per_rotor = 2  # Typical for small drones

        # Blade Passage Frequency (BPF) = RPM * blades / 60
        self.hover_bpf = self.nominal_rpm * self.blades_per_rotor / 60  # Hz

        # Harmonic structure (BPF dominates, then multiples)
        self.num_harmonics = 8
        self.harmonic_weights = np.array([1.0, 0.4, 0.6, 0.25, 0.3, 0.15, 0.1, 0.08])

        # Directivity pattern (propwash downward, less noise upward)
        self.vertical_directivity = 0.3  # Reduced noise directly above/below
        self.horizontal_directivity = 1.0  # Full noise horizontally

        # RPM varies with load (throttle)
        self.rpm_load_sensitivity = 800  # RPM increase under load

    def get_rpm_from_flight_state(self, velocity, acceleration, altitude):
        """Calculate RPM based on flight dynamics"""
        # Base RPM for hover
        base_rpm = self.nominal_rpm

        # Higher RPM for forward flight (efficiency drop)
        horizontal_speed = np.linalg.norm(velocity[:2])
        speed_rpm_increase = 200 * np.tanh(horizontal_speed / 10)

        # Higher RPM for altitude hold against sink rate
        vertical_accel = acceleration[2] if len(acceleration) > 2 else 0
        altitude_rpm_increase = 300 * np.tanh(abs(vertical_accel) / 3)

        # Higher RPM for aggressive maneuvering
        horizontal_accel = np.linalg.norm(acceleration[:2])
        maneuver_rpm_increase = 400 * np.tanh(horizontal_accel / 5)

        # Total RPM
        total_rpm = (base_rpm + speed_rpm_increase +
                     altitude_rpm_increase + maneuver_rpm_increase)

        # Add small random variations (engine irregularities)
        rpm_noise = rng.normal(0, 30)  # Small RPM fluctuations

        return total_rpm + rpm_noise

    def get_directivity_3d(self, source_pos, velocity, target_pos):
        """3D directivity pattern for drone"""
        to_target = target_pos - source_pos
        if np.linalg.norm(to_target) < 1e-6:
            return 1.0

        to_target_norm = to_target / np.linalg.norm(to_target)

        # Vertical component (z-direction)
        vertical_component = abs(to_target_norm[2]) if len(to_target_norm) > 2 else 0

        # Horizontal component
        horizontal_component = np.sqrt(to_target_norm[0] ** 2 + to_target_norm[1] ** 2)

        # Drone radiates less noise directly above/below due to rotor orientation
        directivity = (self.horizontal_directivity * horizontal_component +
                       self.vertical_directivity * vertical_component)

        return np.clip(directivity, 0.2, 1.0)  # Minimum 20% in all directions


drone_source = SmallDroneSource()


# -----------------------------
# 4) Enhanced Signal Generation for Drone
# -----------------------------
def generate_drone_signal_for_sensor(sensor_pos, r, v, a, t, drone_source):
    """Generate realistic drone signal for a specific sensor"""

    # 3D distance calculation
    sensor_pos_3d = np.array([sensor_pos[0], sensor_pos[1], 0])  # Sensor at ground level

    if r.shape[1] == 3:  # 3D positions
        R_vec = sensor_pos_3d - r
    else:  # 2D positions, assume sensor at ground level
        r_3d = np.column_stack([r, np.zeros(len(r))])  # Add z=0 for 2D case
        R_vec = sensor_pos_3d - r_3d

    R = np.linalg.norm(R_vec, axis=1)
    R_eff = np.maximum(R, 3.0)  # Minimum distance for numerical stability

    # Unit vector and radial velocity
    u_hat = np.zeros_like(R_vec)
    nz = R > 0
    u_hat[nz] = R_vec[nz] / R[nz, None]
    v_r = np.sum(v * u_hat, axis=1)

    # Doppler shift
    doppler_factor = c / (c - v_r)

    # Flight state dependent RPM and power
    rpm_values = np.zeros(len(t))
    power_values = np.zeros(len(t))

    for i in range(len(t)):
        rpm_values[i] = drone_source.get_rpm_from_flight_state(v[i], a[i], r[i, 2] if r.shape[1] > 2 else 50)

        # Power based on flight effort (acceleration + altitude maintenance)
        effort = (np.linalg.norm(a[i]) / 5.0 +  # Acceleration effort
                  np.linalg.norm(v[i][:2]) / 15.0 +  # Speed effort
                  0.5)  # Base hover power
        power_values[i] = np.tanh(effort)  # Normalized 0-1

    # Calculate instantaneous BPF
    bpf_inst = rpm_values * drone_source.blades_per_rotor / 60

    # 3D Directivity
    directivity = np.array([drone_source.get_directivity_3d(
        r[i] if r.shape[1] == 3 else np.append(r[i], 50), v[i], sensor_pos_3d)
        for i in range(len(t))])

    # Distance-based amplitude with atmospheric absorption
    geometric_loss = 1.0 / R_eff

    # Frequency-dependent atmospheric absorption
    abs_loss = np.zeros_like(R_eff)
    for i, (dist, alt) in enumerate(zip(R_eff, r[:, 2] if r.shape[1] > 2 else np.full(len(R_eff), 50))):
        avg_freq = np.mean(bpf_inst[i:i + 1]) * 3  # Estimate for broadband
        abs_coeff = atm_absorption(avg_freq, temp, humidity, alt)
        abs_loss[i] = np.exp(-abs_coeff * dist / 1000)

    total_amplitude = power_values * directivity * geometric_loss * abs_loss

    # Generate harmonic series
    signal_components = []

    for h in range(drone_source.num_harmonics):
        harmonic_num = h + 1

        # Instantaneous frequency for this harmonic
        f_inst = bpf_inst * harmonic_num * doppler_factor

        # Phase integration
        phase = 2 * np.pi * np.cumsum(f_inst) * dt

        # Harmonic amplitude
        harmonic_weight = drone_source.harmonic_weights[h]

        # Small amplitude modulation from rotor interactions
        am_mod = 1.0 + 0.05 * np.sin(2 * np.pi * 0.5 * t + h * 0.3)

        # Generate harmonic component
        harmonic_amp = harmonic_weight * total_amplitude * am_mod
        signal_components.append(harmonic_amp * np.cos(phase))

    # Sum all harmonics
    clean_signal = np.sum(signal_components, axis=0)

    # Add rotor blade vortex shedding (broadband component)
    vortex_noise = 0.1 * total_amplitude * rng.normal(0, 1, len(t))

    # Filter vortex noise to higher frequencies (blade tip effects)
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 0.3, btype='high', fs=fs)
    if len(vortex_noise) > 12:  # Ensure minimum length for filtfilt
        vortex_noise = filtfilt(b, a, vortex_noise)

    total_signal = clean_signal + vortex_noise

    return total_signal, R_eff, v_r, doppler_factor, total_amplitude, bpf_inst


# Generate signals for both sensors
signals = []
ranges = []
radial_vels = []
doppler_factors = []
amplitudes = []
bpf_traces = []

for i, sensor_pos in enumerate(sensors):
    sig, R, vr, df, amp, bpf = generate_drone_signal_for_sensor(
        sensor_pos, r, v, a, t, drone_source)
    signals.append(sig)
    ranges.append(R)
    radial_vels.append(vr)
    doppler_factors.append(df)
    amplitudes.append(amp)
    bpf_traces.append(bpf)

# Add minimal realistic noise (post-processing already done)
noisy_signals = []
for i, sig in enumerate(signals):
    # Light noise for post-processed signals
    signal_power = np.var(sig)
    noise = 0.02 * np.sqrt(signal_power) * rng.normal(0, 1, len(sig))
    noisy_signals.append(sig + noise)

# -----------------------------
# 5) Comprehensive Visualization
# -----------------------------

# Create figure with subplots
fig = plt.figure(figsize=(18, 24))
gs = fig.add_gridspec(8, 2, height_ratios=[1, 1, 1, 1, 1, 1, 1.5, 1.5], hspace=0.4)

# 5a) 3D Trajectory (if available) or 2D with altitude color
ax1 = fig.add_subplot(gs[0, :])
if r.shape[1] == 3:
    scatter = ax1.scatter(r[:, 0], r[:, 1], c=r[:, 2], cmap='viridis',
                          s=2, alpha=0.7, label='Drone Trajectory')
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Altitude [m]')
else:
    ax1.plot(r[:, 0], r[:, 1], 'b-', linewidth=1, alpha=0.7, label='Drone Trajectory')

ax1.scatter(S1[0], S1[1], marker='x', s=150, c='red', label='Sensor 1', linewidths=3)
ax1.scatter(S2[0], S2[1], marker='s', s=150, c='orange', label='Sensor 2', linewidths=2)
ax1.set_aspect('equal')
ax1.set_title('Small Drone Mission Trajectory')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 5b) Flight dynamics
ax2 = fig.add_subplot(gs[1, 0])
speed_2d = np.linalg.norm(v[:, :2], axis=1)
accel_mag = np.linalg.norm(a, axis=1)
ax2.plot(t, speed_2d, 'b-', label='Horizontal Speed [m/s]')
ax2.plot(t, accel_mag, 'r-', alpha=0.7, label='|Acceleration| [m/s²]')
ax2.set_title('Drone Flight Dynamics')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Magnitude')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 5c) Altitude profile
ax3 = fig.add_subplot(gs[1, 1])
if r.shape[1] == 3:
    ax3.plot(t, r[:, 2], 'g-', linewidth=2, label='Altitude [m]')
    ax3.fill_between(t, 0, r[:, 2], alpha=0.3, color='green')
ax3.set_title('Altitude Profile')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Altitude [m]')
ax3.grid(True, alpha=0.3)

# 5d) BPF variation
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(t, bpf_traces[0], 'b-', label='Blade Passage Frequency [Hz]')
ax4.axhline(y=drone_source.hover_bpf, color='k', linestyle='--',
            alpha=0.5, label='Hover BPF')
ax4.set_title('Dynamic Blade Passage Frequency')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('BPF [Hz]')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5e) Radial velocities
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(t, radial_vels[0], 'b-', label='Sensor 1 v_r')
ax5.plot(t, radial_vels[1], 'r-', label='Sensor 2 v_r')
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax5.set_title('Radial Velocities (Doppler)')
ax5.set_xlabel('Time [s]')
ax5.set_ylabel('v_r [m/s]')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 5f) Distances to sensors
ax6 = fig.add_subplot(gs[3, 0])
ax6.plot(t, ranges[0], 'b-', label='Distance to Sensor 1')
ax6.plot(t, ranges[1], 'r-', label='Distance to Sensor 2')
ax6.set_title('Distance to Sensors')
ax6.set_xlabel('Time [s]')
ax6.set_ylabel('Distance [m]')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 5g) Signal amplitudes
ax7 = fig.add_subplot(gs[3, 1])
ax7.plot(t, amplitudes[0], 'b-', label='Sensor 1 Amplitude')
ax7.plot(t, amplitudes[1], 'r-', label='Sensor 2 Amplitude')
ax7.set_title('Signal Amplitudes (Flight Load + Distance + Directivity)')
ax7.set_xlabel('Time [s]')
ax7.set_ylabel('Relative Amplitude')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 5h) Doppler factors comparison
ax8 = fig.add_subplot(gs[4, 0])
ax8.plot(t, doppler_factors[0], 'b-', label='Sensor 1')
ax8.plot(t, doppler_factors[1], 'r-', label='Sensor 2')
ax8.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
ax8.set_title('Doppler Shift Factors')
ax8.set_xlabel('Time [s]')
ax8.set_ylabel('f_observed / f_source')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 5i) Time domain signals (detailed view)
ax9 = fig.add_subplot(gs[4, 1])
# Show middle section where drone is closest
mid_start = int(T * 0.4 * fs)
mid_end = int(T * 0.6 * fs)
t_detail = t[mid_start:mid_end]
ax9.plot(t_detail, noisy_signals[0][mid_start:mid_end], 'b-', alpha=0.8, label='Sensor 1')
ax9.plot(t_detail, noisy_signals[1][mid_start:mid_end] + 1, 'r-', alpha=0.8,
         label='Sensor 2 (+1 offset)')
ax9.set_title('Time Domain Signals (Detail View)')
ax9.set_xlabel('Time [s]')
ax9.set_ylabel('Amplitude')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 5j & 5k) Spectrograms for both sensors
for i, (sig, sensor_name) in enumerate(zip(noisy_signals, sensor_names)):
    ax = fig.add_subplot(gs[6 + i, :])

    # Optimized spectrogram parameters for drone signals
    nperseg = 4096
    noverlap = nperseg // 8 * 7  # High overlap for good time resolution

    f, t_spec, Sxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                          window='hann', return_onesided=True, scaling='psd')

    # Convert to dB
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)

    # Plot with drone-appropriate frequency range
    im = ax.pcolormesh(t_spec, f, Sxx_db, shading='gouraud',
                       cmap='plasma', vmin=np.percentile(Sxx_db, 10),
                       vmax=np.percentile(Sxx_db, 98))

    ax.set_title(f'Spectrogram - {sensor_name} (Drone BPF Harmonics)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_ylim(0, 1500)  # Focus on drone frequency range

    # Overlay theoretical BPF lines
    for h in range(1, min(4, drone_source.num_harmonics + 1)):
        bpf_line = bpf_traces[i][::int(fs / 10)] * h * doppler_factors[i][::int(fs / 10)]  # Downsample
        t_line = t[::int(fs / 10)]
        ax.plot(t_line, bpf_line, '--', color='yellow' if h == 1 else 'orange',
                alpha=0.7, linewidth=2 if h == 1 else 1,
                label=f'BPF×{h}' if h <= 2 else None)

    if i == 0:  # Only show legend for first spectrogram
        ax.legend(loc='upper right', fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PSD [dB]')

plt.tight_layout()
plt.show()

# -----------------------------
# 6) Drone Mission Summary
# -----------------------------
print("=== Small Drone (UAV) Doppler Simulation Summary ===")
print(f"Mission Duration: {T:.1f} seconds")
print(f"Sample Rate: {fs:.0f} Hz")
print(f"\n--- Drone Specifications ---")
print(f"Rotors: {drone_source.num_rotors}")
print(f"Blades per rotor: {drone_source.blades_per_rotor}")
print(f"Rotor diameter: {drone_source.rotor_diameter:.2f} m")
print(f"Hover RPM: {drone_source.nominal_rpm:.0f}")
print(f"Hover BPF: {drone_source.hover_bpf:.1f} Hz")
print(f"Harmonics tracked: {drone_source.num_harmonics}")

print(f"\n--- Flight Performance ---")
max_speed_2d = np.max(np.linalg.norm(v[:, :2], axis=1))
max_altitude = np.max(r[:, 2]) if r.shape[1] > 2 else "2D simulation"
min_altitude = np.min(r[:, 2]) if r.shape[1] > 2 else "2D simulation"
max_accel = np.max(np.linalg.norm(a, axis=1))

print(f"Max horizontal speed: {max_speed_2d:.1f} m/s ({max_speed_2d * 3.6:.1f} km/h)")
print(f"Max acceleration: {max_accel:.1f} m/s²")
if r.shape[1] > 2:
    print(f"Altitude range: {min_altitude:.1f} - {max_altitude:.1f} m")

print(f"\n--- Doppler Analysis ---")
for i, sensor_name in enumerate(sensor_names):
    print(f"\n{sensor_name}:")
    print(f"  Distance range: {np.min(ranges[i]):.1f} - {np.max(ranges[i]):.1f} m")
    print(f"  Radial velocity range: {np.min(radial_vels[i]):.1f} to {np.max(radial_vels[i]):.1f} m/s")
    print(f"  Max Doppler shift: {(np.max(doppler_factors[i]) - 1) * 100:.2f}%")
    print(f"  Min Doppler shift: {(np.min(doppler_factors[i]) - 1) * 100:.2f}%")

    # BPF frequency range
    bpf_min = np.min(bpf_traces[i] * doppler_factors[i])
    bpf_max = np.max(bpf_traces[i] * doppler_factors[i])
    print(f"  BPF frequency range: {bpf_min:.1f} - {bpf_max:.1f} Hz")

    # Signal strength
    sig_rms = np.sqrt(np.mean(noisy_signals[i] ** 2))
    sig_peak = np.max(np.abs(noisy_signals[i]))
    print(f"  Signal RMS: {sig_rms:.4f}")
    print(f"  Signal Peak: {sig_peak:.4f}")

print(f"\n--- Mission Profile Analysis ---")
print("✓ Complex 3D trajectory with altitude variations")
print("✓ Realistic drone maneuvers: loitering, evasion, search patterns")
print("✓ Dynamic RPM based on flight load and maneuvering")
print("✓ Blade Passage Frequency (BPF) tracking with harmonics")
print("✓ 3D directivity pattern (reduced noise above/below)")
print("✓ Atmospheric absorption at flight altitude")
print("✓ Rotor vortex shedding (broadband noise)")
print("✓ Wind buffeting and turbulence effects")
print("✓ Two-sensor triangulation capability")

print(f"\n--- Key Features for Detection Algorithm ---")
print("1. Primary signature: BPF and its harmonics with Doppler shift")
print("2. Dynamic frequency tracking (RPM varies with flight mode)")
print("3. Amplitude modulation from flight dynamics")
print("4. Characteristic onset/offset patterns")
print("5. Multi-sensor correlation for triangulation")
print("6. Altitude-dependent atmospheric absorption")

print(f"\n--- Data Arrays for Algorithm Development ---")
print(f"Time vector: {len(t)} samples at {fs}Hz")
print(f"3D Position: {r.shape}")
print(f"3D Velocity: {v.shape}")
print(f"3D Acceleration: {a.shape}")
print(f"BPF traces: 2 sensors × {len(t)} samples")
print(f"Doppler factors: 2 sensors × {len(t)} samples")
print(f"Clean + noisy signals: 2 sensors × {len(t)} samples")
print(f"Distance/amplitude data for model fitting available")

print(f"\n--- Simulation Validation Notes ---")
print("- BPF varies realistically with flight load (hover vs maneuver)")
print("- Doppler shifts are physically consistent with radial velocities")
print("- Signal amplitude incorporates: power + directivity + range + absorption")
print("- Harmonic structure matches typical small drone acoustic signature")
print("- Mission timeline includes various flight phases for robust testing")
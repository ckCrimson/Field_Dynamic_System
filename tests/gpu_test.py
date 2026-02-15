import jax

# 1. Check the primary backend (CPU, GPU, or TPU)
print(f"Primary JAX Backend: {jax.default_backend().upper()}")

# 2. List the specific hardware devices JAX has access to
devices = jax.devices()
print(f"Available Devices ({len(devices)}):")
for d in devices:
    print(f"  - {d.device_kind} (ID: {d.id})")
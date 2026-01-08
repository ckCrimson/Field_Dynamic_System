"""
Performance Benchmark for State Encoders.
Compares Naive Loop Encoding vs. Optimized Batch Encoding.
"""
import time
import jax.numpy as jnp
from src.field_dynamic_system.core.state.encoding import VectorEncoding, BitMaskingEncoding
from src.field_dynamic_system.core.state.encoding import VectorState,AbstractState

N_ITEMS = 100_000

def benchmark_vector_encoding():
    # ... setup ...
    states = [VectorState(values=(i * 0.1, i * 0.2)) for i in range(N_ITEMS)]
    encoder = VectorEncoding(dim=2)

    # A. Slow Way (Manually looping)
    start = time.time()
    # The user forces single calls
    _ = [encoder.encode(s) for s in states]
    slow_time = time.time() - start
    print(f"[Old Way] Manual Loop:   {slow_time:.4f}s")

    # B. Fast Way (Passing the list)
    start = time.time()
    # The user just passes the list
    grid_data = encoder.encode(states)
    fast_time = time.time() - start
    print(f"[New Way] List Input:    {fast_time:.4f}s")

    # Stats
    speedup = slow_time / fast_time
    print(f"🚀 Speedup: {speedup:.1f}x")

def benchmark_discrete_encoding():
    print(f"\n--- 2. Benchmarking Discrete Encoding ({N_ITEMS} items) ---")

    # Setup
    rock = AbstractState(name="Rock", properties={})
    paper = AbstractState(name="Paper", properties={})
    scissors = AbstractState(name="Scissors", properties={})
    universe = [rock, paper, scissors]

    # Create a random pattern of states
    print("Generating objects...")
    # Just repeating the pattern to reach N_ITEMS
    states = [universe[i % 3] for i in range(N_ITEMS)]
    encoder = BitMaskingEncoding(universe)

    # A. Slow Way (Loop)
    start = time.time()
    _ = [encoder.encode(s) for s in states]
    slow_time = time.time() - start
    print(f"[Old Way] Loop Encoding:   {slow_time:.4f}s")

    # B. Fast Way (Batch)
    start = time.time()
    grid_data = encoder.encode(states)
    fast_time = time.time() - start
    print(f"[New Way] Batch Encoding:  {fast_time:.4f}s")

    # Stats
    speedup = slow_time / fast_time
    print(f"🚀 Speedup: {speedup:.1f}x")

    # Verify JAX Data
    assert grid_data.shape == (N_ITEMS, 1)
    print("✅ JAX Shape Verified")

if __name__ == "__main__":
    benchmark_vector_encoding()
    benchmark_discrete_encoding()
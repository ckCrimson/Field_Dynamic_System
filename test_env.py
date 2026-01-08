import jax
import jax.numpy as jnp
# This checks if the 'src' layout is working:
import field_dynamic_system

def smoke_test():
    print(f"Package Location: {field_dynamic_system.__file__}")
    print(f"JAX Version: {jax.__version__}")
    print(f"Device: {jax.devices()[0]}")

    # Matrix Math Test
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    y = jax.random.normal(key, (1000, 1000))
    result = jnp.dot(x, y).block_until_ready()

    print("SUCCESS: Environment and Source Layout are perfect.")

if __name__ == "__main__":
    smoke_test()
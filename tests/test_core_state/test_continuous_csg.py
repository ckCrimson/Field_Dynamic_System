"""
Logic Test for Continuous Space CSG (Union/Intersection).
Verifies: Hypercube (Square) vs Hypersphere (Circle).
"""
import jax.numpy as jnp
from src.field_dynamic_system.core.state.encoding import VectorEncoding
from src.field_dynamic_system.core.state.continous  import HypercubeSpace, HypersphereSpace
from src.field_dynamic_system.core.state import VectorState


def test_csg_logic():
    print("\n--- Testing Continuous CSG Logic (Square vs Circle) ---")

    # 1. Setup Encoders/States
    enc = VectorEncoding(dim=2)

    # 2. Define Shapes
    # Square: Side 4 (from -2 to 2)
    square = HypercubeSpace(
        low=jnp.array([-2.0, -2.0]),
        high=jnp.array([2.0, 2.0]),
        _encoding=enc
    )

    # Circle: Radius 2.5
    circle = HypersphereSpace(
        center=jnp.array([0.0, 0.0]),
        radius=2.5,
        _encoding=enc
    )

    # 3. Define Test Points
    # P1: Center (In Both)
    p_center = jnp.array([0.0, 0.0])

    # P2: Corner of Square (1.9, 1.9)
    # Dist = sqrt(1.9^2 + 1.9^2) = ~2.68 (Outside Circle, Inside Square)
    p_square_only = jnp.array([1.9, 1.9])

    # P3: Edge of Circle (2.4, 0)
    # X=2.4 is > 2.0 (Outside Square, Inside Circle)
    p_circle_only = jnp.array([2.4, 0.0])

    # P4: Far away (3, 3) (Outside Both)
    p_far = jnp.array([3.0, 3.0])

    print("Shapes Defined.")

    # --- TEST 1: UNION (OR) ---
    print("\n[Testing UNION (Square U Circle)]")
    union_space = square.union(circle)

    # Logic Checks
    assert union_space.contains(p_center) == True, "Center should be in Union"
    assert union_space.contains(p_square_only) == True, "Square corner should be in Union"
    assert union_space.contains(p_circle_only) == True, "Circle edge should be in Union"
    assert union_space.contains(p_far) == False, "Far point should NOT be in Union"

    # Projection Check (Projecting the far point)
    # Should snap to the closer shape (likely the circle edge or square corner)
    proj = union_space.project(p_far)
    print(f"Projected Far Point {p_far} -> {proj}")
    # Verify the projection is now considered "Inside"
    assert union_space.contains(proj) == True, "Projected point must be valid"
    print("✅ Union Logic Passed")

    # --- TEST 2: INTERSECTION (AND) ---
    print("\n[Testing INTERSECTION (Square ∩ Circle)]")
    inter_space = square.intersection(circle)

    # Logic Checks
    assert inter_space.contains(p_center) == True, "Center should be in Intersection"
    assert inter_space.contains(p_square_only) == False, "Square corner (2.68 > 2.5) should NOT be in Intersection"
    assert inter_space.contains(p_circle_only) == False, "Circle edge (2.4 > 2.0) should NOT be in Intersection"
    assert inter_space.contains(p_far) == False, "Far point should NOT be in Intersection"

    # Projection Check (The Complex POCS Algorithm)
    # We project 'p_square_only'. It is inside Square but outside Circle.
    # The intersection logic should push it inside the circle radius.
    proj_inter = inter_space.project(p_square_only)

    dist_origin = jnp.linalg.norm(proj_inter)
    print(f"Projected Square Corner {p_square_only} -> {proj_inter}")
    print(f"Distance from origin: {dist_origin:.4f} (Should be <= 2.5)")

    assert inter_space.contains(proj_inter) == True, "Projected intersection point must be valid"
    print("✅ Intersection Logic Passed")


if __name__ == "__main__":
    test_csg_logic()
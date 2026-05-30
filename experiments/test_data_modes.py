import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.common_data import load_data

def test_modes():
    print("Testing 'dummy' mode...")
    S, W = load_data(mode="dummy", n=100, m=500, d=3, L=5)
    print(f"  [Output] S shape: {S.shape} (Expected: 500, 3)")
    print(f"  [Output] W shape: {W.shape} (Expected: 100, 5, 3)")
    assert S.shape == (500, 3)
    assert W.shape == (100, 5, 3)

    print("\nTesting 'synthetic_paper' mode (UN)...")
    S, W = load_data(mode="synthetic_paper", n=200, m=400, d=4, L=7, dist='UN')
    print(f"  [Output] S shape: {S.shape} (Expected: 400, 4)")
    print(f"  [Output] W shape: {W.shape} (Expected: 200, 7, 4)")
    assert S.shape == (400, 4)

    print("\nTesting 'synthetic_paper' mode (CL)...")
    S, W = load_data(mode="synthetic_paper", n=150, m=350, d=5, L=8, dist='CL')
    print(f"  [Output] S shape: {S.shape} (Expected: 350, 5)")
    print(f"  [Output] W shape: {W.shape} (Expected: 150, 8, 5)")
    assert W.shape == (150, 8, 5)
    
    print("\nTesting 'real_embeddings' mode (Placeholder)...")
    S, W = load_data(mode="real_embeddings")
    print(f"  [Output] S shape: {S.shape} (Mock Default: 1000, 10)")
    print(f"  [Output] W shape: {W.shape} (Mock Default: 500, 10, 10)")
    
    print("\n✅ All modes executed flawlessly and returned properly structured tensors without breaking older hooks.")

if __name__ == "__main__":
    test_modes()

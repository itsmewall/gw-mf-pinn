import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from physics import gwphys_cuda as m
import numpy as np

print("CUDA:", m.info())
A = np.random.randn(4, 8).astype(np.float32)
B = np.random.randn(3, 8).astype(np.float32)
C = m.cosine_scores_gpu(A, B)

# validação
C_ref = (A @ B.T) / (np.linalg.norm(A, axis=1, keepdims=True) * np.linalg.norm(B, axis=1, keepdims=True).T + 1e-9)
err = np.linalg.norm(C - C_ref)
print("C shape:", C.shape, "min/max:", C.min(), C.max())
print("L2 error:", err)

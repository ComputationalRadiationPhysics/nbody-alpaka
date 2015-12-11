# Kernels (first ideas)
## Force matrix (triangle method)
- for i <= j
    - calculate Fij
    - set Fji = -Fij
- for i > j
    - do nothing (Good for CPU, mediocre for OpenMP and bad for GPU)

## Acceleration and velocity kernel
- Ai = sum(j from 0 to n - 1){ Fij }
- Pi += 0.5\*Ai\*dt^2 + Vi\*dt
- Vi += Ai\*dt

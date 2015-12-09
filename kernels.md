# Kernels (first ideas)
## Force matrix (triangle method)
- for i <= j
    - calculate Fij
    - set Fji = -Fij
- for i > j
    - do nothing (Good for CPU, mediocre for OpenMP and bad for GPU)

## Acceleration and velocity kernel
- Ai = sum(j from 0 to n - 1){ Fij }
- Vi += Ai\*dt
- Pi += (Vi\*dt)/2

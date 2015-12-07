# Kernels
## Force matrix (triangle)
- for i <= j
    - calculate Fij
    - set Fji = -Fij
- for i > j
    - do nothing

## Acceleration and velocity kernel
- Ai = sum(j from 0 to n - 1){ Fij }
- Vi += Ai\*dt
- Pi += (Vi\*dt)/2

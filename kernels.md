# Kernels (first ideas)
## Force matrix formula
- Fij = ( G \* mi \* mj) \* (Pj - Pi) / abs(Pj - Pi)^3

## Force matrix (triangle method)
- for i <= j
    - calculate Fij
    - set Fji = -Fij
- for i > j
    - do nothing (Good for CPU, mediocre for OpenMP and bad for GPU)

## Acceleration and velocity kernel
- Acceleration of body i
    - Ai = 1/mi \* sum(j from 0 to n - 1){ Fij }
- Position of body i
    - Pi += 0.5\*Ai\*dt^2 + Vi\*dt
- Velocity of body i
    - Vi += Ai\*dt

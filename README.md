# Music Genre Pattern Detection Using Linear Algebra
**PES University | UE24MA241B — Linear Algebra and Its Applications**

## Problem Statement
Detecting music genre patterns from raw audio features using a complete linear algebra pipeline — no machine learning used.

## Pipeline
1. Real-World Data → Matrix Representation
2. Matrix Simplification (RREF / Gaussian Elimination)
3. Structure of the Space (Rank & Nullity)
4. Remove Redundancy (Linear Independence / Basis)
5. Orthogonalization (Gram–Schmidt)
6. Projection onto Feature Subspace
7. Least Squares Approximation
8. Pattern Discovery (Eigenvalues & Eigenvectors)
9. System Simplification (Diagonalization) → Genre Cluster Plot

## Dataset
GTZAN Genre Dataset — https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Download `features_30_sec.csv` and place it in the project folder.

## How to Run
```bash
pip install numpy pandas matplotlib
python mini.py
```

## Output
- 2D Genre Cluster Map showing songs grouped by genre in eigenspace
- Scree Plot showing variance explained by each eigenvector

## Tools
Python, NumPy, Pandas, Matplotlib

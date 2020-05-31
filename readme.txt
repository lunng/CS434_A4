How to run:

- Make sure dependencies are installed - numpy, pandas, matplotlib, and seaborn.

To run kmeans without PCA:
- python main.py --kmeans 1

To run kmeans with PCA:
- python main.py --kmeans 1 --pca 1

To change retain ratio:
- python main.py --kmeans 1 --pca 1 --pca_retain_ratio [x]
  where x is the desired ratio

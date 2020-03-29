# IVHD-CUDA

This is the *official* implementation of the **IVHD-CUDA** by the original authors, which is used in  visualize large-scale and high-dimensional data "Fast visualization of large multidimensional data inGPU/CUDA environment, by embedding of sparsekNNgraphs". The package also contains a brief description of how to setup FAISS library used for kNN graph generation.

Contact person: Bartosz Minch (minch@agh.edu.pl), Witold Dzwinel (dzwinel@agh.edu.pl).

# Prerequisites

- CUDA (8.0+), NVCC, cuBlas,
- GCC,
- CMAKE,
- FAISS,
- Python (3.6+),
 
It is recommended to use Mac/Linux OS (FAISS library doesn't support Windows for now).

# Compilation

1. Clone this repository.
2. Init and update git submodules:
    - `git submodule init`
    - `git submodule update`
3. Compile using CMake (easily importable to VS Code or CLion).

# Run

Command: `./ivhd dataset_file knn_file it rn exp alg seed`

- dataset_file - dataset in .csv format,
- knn_file - knn graph produced by FAISS library,
- it - number of iterations,
- rn - number of random neighbors loaded from kNN graph,
- exp - name of experiments (where results will be saved),
- alg - optimization algorithms available for IVHD method - both CPU and GPU ("nesterov", "adadelta_sync", "adadelta_async", "cuda_ab", "cuda_nesterov", "cuda_adadelta", "cuda_adam"),

Output:
- `result_file` containing embedding output,
- `error_file` containing errors in next time steps,

# Visualization

- Calculating kNN metric used in paper desribed above:
  - `python ./knn_metric.py result_file number_of_neighbors`

- Plotting error:
  - `python ./plot_error.py error_file`

- Drawing result on 2-D plane:
  - `python ./draw.py result_file`
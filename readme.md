# Sampling Clustering
#### Ching Tarn, Yinan Zhang and Ye Feng. (Jun 2018)

The paper is available at [arxiv](https://arxiv.org/abs/1806.08245).

We have provided scripts to download datasets like MNIST.
Other datasets pre-processed by Sohil Shah can be found at [Google Drive](https://drive.google.com/drive/folders/1vN4IpmjJvRngaGkLSyKVsPaoGXL02mFf).

## Usage
### clone the repository **recursively**
```sh
git clone git@github.com:ctarn/sampling-clustering.git --recursive
```

### requirements
- python 3
- numpy
- sklearn
- networkx
- matplotlib (optional)
- keras (optional, used to download datasets)

### download datasets and generate knn graph
```sh
cd data
./download.py pendigits [<name> ...]
./knn.py --k=16 pendigits/ [<dir> ...]
```

You can also generate a 2d layout of the graph using t-SNE. (optional)
```sh
./visualize.py pendigits/
```

### run sampling clustering
```sh
./demo.py data/pendigits/16-nn.npy --smooth --balance
```


**note:**
We refactored the code, and thus the results may be sightly inconsistent with the paper.
If you want to reproduce the results reported in the paper, please use [the original implementation](https://github.com/ctarn/sampling-clustering/tree/arxiv).

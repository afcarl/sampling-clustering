We propose an efficient graph-based divisive cluster analysis approach called _sampling clustering_.
It constructs a lite informative dendrogram by recursively dividing a graph into subgraphs.
In each recursive call, a graph is sampled first with a set of vertices being removed to disconnect latent clusters,
then condensed by adding edges to the remaining vertices to avoid graph fragmentation caused by vertex removals.
We also present some sampling and condensing methods and discuss the effectiveness in this paper.
Our implementations run in linear time and achieve outstanding performance on various types of datasets.
Experimental results show that they outperform state-of-the-art clustering algorithms with significantly less computing resources requirements.

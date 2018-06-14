#!/usr/bin/env python3

import os
import sys
import argparse
import pickle
import math

import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
from sklearn import metrics

from dendrogram import Dendrogram


class SamplingClustering:
    def __init__(self, sampler='indegree', r=0.2, condenser='sp', t=16):
        self.sampler = sampler
        self.r = r
        self.condenser = condenser
        self.t = t

    def fit(self, graph, sm=None, gt=None):
        if type(graph) is not nx.DiGraph:
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(np.arange(len(graph)))
            for node, nbrs in enumerate(graph):
                self.graph.add_weighted_edges_from([(node, nbr, idx) for idx, nbr in enumerate(nbrs)])
        else:
            self.graph = graph
        self.sm = sm
        self.gt = gt
        self.labels = -np.ones(len(self.graph.nodes), dtype=int)
        self.atree = nx.DiGraph()  # association
        self.atree.add_nodes_from(self.graph)

        graph = condense(graph=self.graph, t=self.t, measure=('sn', 'sp'))
        subgraphs = [graph.subgraph(c).copy() for c in nx.strongly_connected_components(graph)]
        self.dendrogram = Dendrogram([self.cluster(graph=subgraph) for subgraph in subgraphs])
        self.labels = self.dendrogram.flatten()
        return self

    def cluster(self, graph, **kwargs):
        removed = sample(graph=graph, rate=self.r, measure=self.sampler)
        if len(removed) == 0 or len(removed) == len(graph.nodes):
            head = self.associate(list(graph.nodes))
            return [int(node) for node in list(nx.descendants(self.atree, head)) + [head]]
        else:
            self.associate(removed, graph)
            graph.remove_nodes_from(removed)
            graph = condense(graph=graph, t=self.t, measure=self.condenser, sm=self.sm)
            subgraphs = [graph.subgraph(c).copy() for c in nx.strongly_connected_components(graph)]
            return [self.cluster(subgraph) for subgraph in subgraphs]

    def associate(self, nodes, graph=None):
        if graph is None:
            head = nodes.pop()
            self.atree.add_edges_from([(head, node) for node in nodes])
            return head
        for node in nodes:
            for u, v in nx.bfs_edges(graph, node):
                if v not in nodes:
                    self.atree.add_edge(v, node)
                    break
            else:
                print('error: failed to associate %v' % node)

    def smooth_labels(self, labels=None):
        return smooth_labels((self.labels if labels is None else labels), self.graph)

    def smooth_dendrogram(self, dendrogram=None):
        return smooth_dendrogram((self.dendrogram if dendrogram is None else dendrogram), self.graph)


def smooth_labels(labels, graph):
    local_common_label = lambda node: np.bincount(labels[list(graph.succ[node]) + [node]]).argmax()
    return np.array([local_common_label(node) for node in range(len(labels))])


def smooth_dendrogram(dendrogram, graph):
    labels, leaves = dendrogram.flatten(return_leaves=True)
    labels = smooth_labels(labels, graph)
    for n, leaf in enumerate(leaves):
        leaf.clear()
        leaf += [int(node) for node in np.arange(len(labels))[labels == n]]
    dendrogram.format()


def sample(graph, rate, measure, **kwargs):
    if measure == 'indegree':
        key = graph.in_degree
    elif measure == 'mutual':
        mutual = lambda u: len([v for v in graph.succ[u] if graph.has_edge(v, u)])
        key = [(node, mutual(node)) for node in graph.nodes]
    elif measure == 'random':
        return set([node for node in graph.nodes if np.random.random() < rate])
    n = int(len(graph.nodes) * rate)
    thres = np.partition(np.array([v for _, v in key]), n)[n]
    return set([node for node, v in key if v < thres])


def condense(graph, t, measure, **kwargs):
    measure = (measure,) if type(measure) == str else measure
    dist = {}
    if 'sp' in measure:  # shortest path
        sp = lambda node: {nbr: idx for idx, (_, nbr) in zip(range(t), nx.bfs_edges(graph, node))}
        dist['sp'] = {node: sp(node) for node in graph.nodes}
    if 'sn' in measure:  # shared neighbor
        iou = lambda u, v: -(graph.out_degree[u] + graph.out_degree[v] + 2) / len(set(graph.succ[u]) | set(graph.succ[v]) | {u, v})
        sn = lambda node: {nbr: iou(node, nbr) for _, nbr in nx.dfs_edges(graph, node, depth_limit=2)}
        dist['sn'] = {node: sn(node) for node in graph.nodes}
    if 'sm' in measure:  # similarity matrix
        matrix = kwargs['sm']
        sm = lambda node: {nbr: matrix(node, nbr) for _, nbr in nx.dfs_edges(graph, node, depth_limit=2)}
        dist['sm'] = {node: sm(node) for node in graph.nodes}

    get_nbrs = lambda node: np.unique(np.concatenate([list(dist[s][node].keys()) for s in measure]))
    get_key = lambda u, v: [dist[s][u].get(v, math.inf) for s in measure]
    nbrs = {node: sorted(get_nbrs(node), key=lambda nbr: get_key(node, nbr)) for node in graph.nodes}

    graph_ = nx.DiGraph()
    graph_.add_nodes_from(graph)
    for node in graph.nodes:
        graph_.add_weighted_edges_from([(node, nbr, idx) for idx, nbr in zip(range(t), nbrs[node])])
    return graph_


def draw_labels(pos, labels, path=None, sort=True):
    plt.figure()
    plt.axis('off')
    pos = [(pos[labels==label, 0], pos[labels==label, 1]) for label in np.unique(labels)]
    if sort:
        pos.sort(key=lambda s: (np.mean(s)))
    [plt.scatter(x, y, s=4) for x, y in pos]
    plt.savefig(path) if path else plt.show()
    plt.close()


def evaluate_clustering(gt, labels, name=None):
    score = {
        'Name': name,
        'N': len(np.unique(labels)),
        'N_': len(np.unique(gt)),
        'ARI': metrics.adjusted_rand_score(gt, labels),
        'AMI': metrics.adjusted_mutual_info_score(gt, labels),
        'NMI': metrics.normalized_mutual_info_score(gt, labels),
    }
    print('%(Name)s: N=%(N)4d/%(N_)4d, ARI=%(ARI)0.4f, AMI=%(AMI)0.4f, NMI=%(NMI)0.4f' % score)
    return score


def main(args):
    data_dir = os.path.dirname(args.data)
    name = os.path.basename(data_dir) + '_' + os.path.splitext(os.path.basename(args.data))[0]
    load = lambda file: np.load(os.path.join(data_dir, file)) if os.path.isfile(os.path.join(data_dir, file)) else None
    X = load('X.npy')
    gt = load('Y.npy')
    graph = load(os.path.basename(args.data))
    layout = load('layout.npy')

    tmp = os.path.join('tmp', name)
    os.makedirs(tmp) if not os.path.isdir(tmp) else None
    path = lambda file: os.path.join(tmp, file)

    sm = lambda a, b: euclidean(X[a], X[b])
    sc = SamplingClustering(sampler=args.sampler, r=args.r, condenser=args.condenser, t=args.t).fit(graph, sm=sm, gt=gt)

    def evaluate(dendrogram, name=None):
        if args.log:
            sys.stdout = open(path('t%d%s.log' % (args.t, ('_' + name if name else ''))), 'w')
        title = ('SC (%s)' % name) if name else 'SC'
        print(title)
        print(dendrogram)
        labels = dendrogram.flatten()
        _path = lambda ext: path('t%d%s.%s' % (args.t, ('_' + name if name else ''), ext))
        score = evaluate_clustering(sc.gt, labels, name=title) if sc.gt is not None else None
        pickle.dump({'dendrogram': dendrogram, 'score': score}, open(_path('pkl'), 'wb'))
        np.save(_path('npy'), labels)
        dendrogram.draw(layout, path=_path('gif')) if args.visual else None
        draw_labels(layout, labels, path=_path('png')) if args.visual else None
        if args.log:
            sys.stdout.close()

    evaluate(sc.dendrogram)
    if args.smooth:
        [sc.smooth_dendrogram() for i in range(16)]
        evaluate(sc.dendrogram, 'smooth')

    if gt is not None:
        draw_labels(layout, gt, path=path('gt.png')) if args.visual else None

        sc.dendrogram.merge(n=len(np.unique(gt)), balance=args.balance)
        evaluate(sc.dendrogram, 'merge')
        if args.smooth:
            [sc.smooth_dendrogram() for i in range(16)]
            evaluate(sc.dendrogram, 'merge_smooth')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Sampling Clustering')
    argparser.add_argument('--data', type=str, metavar='<path>', required=True)
    argparser.add_argument('--sampler', type=str, metavar='<measure>', default='indegree')
    argparser.add_argument('--r', type=float, default=0.2)
    argparser.add_argument('--condenser', type=str, metavar='<measure>', default='sp')
    argparser.add_argument('--t', type=int, default=16)
    argparser.add_argument('--balance', action='store_true')
    argparser.add_argument('--smooth', action='store_true')
    argparser.add_argument('--log', action='store_true')
    argparser.add_argument('--visual', action='store_true')
    args = argparser.parse_args()
    main(args)

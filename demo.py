#!/usr/bin/env python3

import os
import sys
import argparse
import pickle

import numpy as np
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
from sklearn import metrics

from clearn.cluster import SamplingClustering
from clearn.cluster.sampler import RandomSampler, IndegreeSampler, MutualNeighborSampler, MinCoverSampler
from clearn.cluster.condenser import BFSCondenser, SharedNeighborCondenser, StaticCondenser


Samplers = {
    'random': RandomSampler,
    'indegree': IndegreeSampler,
    'mutual': MutualNeighborSampler,
    'cover': MinCoverSampler,
}


Condensers = {
    'bfs': BFSCondenser,
    'shared': SharedNeighborCondenser,
    'static': StaticCondenser,
}


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
    results = {
        'Name': name,
        'N': len(np.unique(labels)),
        'N_': len(np.unique(gt)),
        'ARI': metrics.adjusted_rand_score(gt, labels),
        'AMI': metrics.adjusted_mutual_info_score(gt, labels),
        'NMI': metrics.normalized_mutual_info_score(gt, labels),
    }
    print('%(Name)s: N=%(N)4d/%(N_)4d, ARI=%(ARI)0.4f, AMI=%(AMI)0.4f, NMI=%(NMI)0.4f' % results)
    return results


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
    path = lambda items, ext: os.path.join(tmp, '%s.%s' % ('_'.join(items), ext))

    sampler = Samplers[args.sampler](rate=args.rate)
    condenser = Condensers[args.condenser](size=args.size, depth=args.depth, metrix=lambda a, b: euclidean(X[a], X[b]))
    pre_condenser = SharedNeighborCondenser(size=args.size, depth=args.depth)
    sc = SamplingClustering(sampler=sampler, condenser=condenser, pre_condenser=pre_condenser).fit(graph)

    prefix = [args.sampler, 'r' + str(args.rate), args.condenser, 's' + str(args.size), 'd' + str(args.depth)]
    def evaluate(dendrogram, type=None):
        items = (prefix + [type]) if type is not None else prefix
        if args.log:
            sys.stdout = open(path(items, 'log'), 'w')
        title = ('SC (%s)' % type) if type else 'SC'
        print(title)
        print(dendrogram)
        labels = dendrogram.flatten()
        eval = evaluate_clustering(gt, labels, name=title) if gt is not None else None
        pickle.dump({'dendrogram': dendrogram, 'eval': eval}, open(path(items, 'pkl'), 'wb'))
        np.save(path(items, 'npy'), labels)
        if args.visual:
            dendrogram.draw(layout, path=path(items, 'gif'))
            draw_labels(layout, labels, path=path(items, 'png'))
        if args.log:
            sys.stdout.close()

    evaluate(sc.dendrogram)
    if args.smooth:
        [sc.smooth() for i in range(16)]
        evaluate(sc.dendrogram, 'smooth')

    if gt is not None:
        draw_labels(layout, gt, path=path(['gt'], 'png')) if args.visual else None

        sc.dendrogram.merge(n=len(np.unique(gt)), balance=args.balance)
        evaluate(sc.dendrogram, 'merge')
        if args.smooth:
            [sc.smooth() for i in range(16)]
            evaluate(sc.dendrogram, 'merge_smooth')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Sampling Clustering')
    argparser.add_argument('data', type=str, metavar='<path>')
    argparser.add_argument('--sampler', type=str, metavar='<measure>', default='indegree')
    argparser.add_argument('--rate', type=float, default=0.2)
    argparser.add_argument('--condenser', type=str, metavar='<measure>', default='bfs')
    argparser.add_argument('--size', type=int, default=16)
    argparser.add_argument('--depth', type=int, default=2)
    argparser.add_argument('--balance', action='store_true')
    argparser.add_argument('--smooth', action='store_true')
    argparser.add_argument('--log', action='store_true')
    argparser.add_argument('--visual', action='store_true')
    args = argparser.parse_args()
    main(args)

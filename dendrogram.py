import collections
import queue

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


is_leaf = lambda node: type(node) is list and np.array([type(child) is not list for child in node]).all()
is_mergeable = lambda node: type(node) is list and np.array([is_leaf(child) for child in node]).all()

def get_descendants(node):
    descendants = [child for child in node if not isinstance(child, collections.Iterable)]
    for child in [child for child in node if isinstance(child, collections.Iterable)]:
        descendants += get_descendants(child)
    return descendants


def get_leaves(tree, pred=False):
    leaves = [(child, tree) if pred else child for child in (tree + [tree]) if is_leaf(child)]
    for child in tree:
        if type(child) is list and not is_leaf(child):
            leaves += get_leaves(child, pred=pred)
    return leaves


def get_mergeable(tree, pred=False):
    mergeable = [(child, tree) if pred else child for child in (tree + [tree]) if is_mergeable(child)]
    for child in tree:
        if type(child) is list and not is_mergeable(child):
            mergeable += get_mergeable(child, pred=pred)
    return mergeable


class Dendrogram:
    def __init__(self, data):
        self.data = data
        self.format()

    def __str__(self):
        def _print(dendrogram, level=0, prefix='', is_last=True):
            lines = [('%s%s%d' % (prefix, '└── ' if is_last else '├── ', len(get_descendants(dendrogram))))]
            branches = [child for child in dendrogram if isinstance(child, collections.Iterable)]
            if len(branches) != 0:
                for child in branches[:-1]:
                    lines += _print(child, level + 1, prefix + ('    ' if is_last else '│   '), is_last=False)
                lines += _print(branches[-1], level + 1, prefix + ('    ' if is_last else '│   '), is_last=True)
            return lines
        return '\n'.join(_print(self.data))

    def descendants(self):
        return get_descendants(self.data)

    def leaves(self, pred=False):
        return get_leaves(self.data, pred=pred)

    def format(self):
        def _clean(node):
            branches = [child for child in node if isinstance(child, collections.Iterable)]
            for branch in branches:
                _clean(branch)
                if len(branch) == 0 or (len(branch) == 1 and not is_leaf(branch)):
                    node += branch
                    node.remove(branch)
        _clean(self.data)

    def merge(self, n, balance=True):
        if balance:
            while len(self.leaves()) > n:
                leaves = self.leaves(pred=True)
                leaves.sort(key=lambda x: len(x[0]))
                node, pred = leaves[0]
                if len(pred) > 1:
                    pred.sort(key=lambda x: len(get_descendants(x)))
                    nbr = pred[1]
                    nbr += node if is_leaf(nbr) else [node]
                    pred.remove(node)
                if len(pred) == 1:
                    pred += pred[0]
                    pred.remove(pred[0])
        else:
            tot = len(get_descendants(self.data))
            while True:
                leaves = sorted(self.leaves(), key=lambda x: -len(x))
                node = sorted(get_mergeable(self.data), key=lambda x: len(get_descendants(x)))[0]
                top_n = sum([len(leaf) for leaf in leaves[:min(len(leaves), n)]])
                if  top_n < tot * 0.8 and len(leaves) + 1 - len(node) >= n:
                    descendants = get_descendants(node)
                    node.clear()
                    node += descendants
                else:
                    break
        return self

    def flatten(self, return_leaves=False):
        leaves = self.leaves()
        labels = -np.ones(sum([len(leaf) for leaf in leaves]), dtype=int)
        for label, leaf in enumerate(leaves):
            labels[leaf] = label
        return (labels, leaves) if return_leaves else labels

    def draw(self, pos, path):
        ims = []
        fig = plt.figure()
        plt.axis('off')
        fifo = queue.Queue()
        fifo.put([self.data])
        while not fifo.empty():
            node = fifo.get()
            divisible = False
            for child in node:
                if isinstance(child, collections.Iterable):
                    fifo.put(child)
                    nodes = get_descendants(child)
                    plt.scatter(pos[nodes, 0], pos[nodes, 1], s=4)
                    divisible = True
            if divisible:
                fig.canvas.draw()
                ims.append(Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
        ims[0].save(path, save_all=True, append_images=ims, duration=1000)
        plt.close()

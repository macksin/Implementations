class Node:
    
    def __init__(self, data, dim, depth, cut=None, block=None, cuts=None):
        self.blockEnd = False
        self.data = data
        self.shape = data.shape
        self.dim = dim
        self.depth = depth
        self.leftNode = None
        self.rightNode = None
        self.cut = cut
        self.block = block
        self.cuts = cuts
        self.results__ = []
        
    def insert(self, cuttoff, dim, block, cuttoffs):
        depth = self.depth + 1
        X = self.data
        left_idx, = np.where(X[:, dim] <= cuttoff)
        data_left = X[left_idx, :]
        if (self.leftNode is None) and (len(left_idx) > 0):
            self.leftNode = Node(data_left, dim=dim, depth=depth, cut=cuttoff, block=block, cuts=cuttoffs)
        elif (len(left_idx) > 0):
            self.leftNode.insert(cuttoff, dim, block=block, cuttoffs=cuttoffs)
            
        right_idx, = np.where(X[:, dim] > cuttoff)
        data_right = X[right_idx, :]
        if (self.rightNode is None) and (len(right_idx) > 0):
            self.rightNode = Node(data_right, dim=dim, depth=depth, cut=cuttoff, block=block, cuts=cuttoffs)
        elif (len(right_idx) > 0):
            self.rightNode.insert(cuttoff, dim, block=block, cuttoffs=cuttoffs)

    def __markNodeEnds(self, tree):
        if tree.leftNode:
            if (tree.leftNode is None) and (tree.rightNode is None):
                tree.blockEnd = True
            else:
                self.__markNodeEnds(tree.leftNode)
                tree.blockEnd = False
        if tree.rightNode:
            if (tree.leftNode is None) and (tree.rightNode is None):
                tree.blockEnd = True
            else:
                self.__markNodeEnds(tree.rightNode)
                tree.blockEnd = False
        if tree.leftNode is None and tree.rightNode is None:
            tree.blockEnd = True

    def __insert_block(self, tree):
        if (tree.blockEnd == True) and (tree.leftNode is None) and (tree.rightNode is None):
            block = (tree.block or 0) + 1
            cuttoffs = np.percentile(tree.data, q=50, axis=0)
            tree.insert(cuttoffs[0], 0, block=block, cuttoffs=cuttoffs)
            tree.insert(cuttoffs[1], 1, block=block, cuttoffs=cuttoffs)
        else:
            if tree.leftNode:
                self.__insert_block(tree.leftNode)
            if tree.rightNode:
                self.__insert_block(tree.rightNode)
            

    def __findNode(self, tree, list_of_results):

        if (tree.leftNode is None and tree.rightNode is None):
            return (tree.data, tree.cuts)

        if tree.leftNode:
            list_of_results.append(self.__findNode(tree.leftNode, list_of_results))
        if tree.rightNode:
            list_of_results.append(self.__findNode(tree.rightNode, list_of_results))

    def __cleanResult(self, listOfResults):
        Result = [e for e in listOfResults if e is not None]
        return Result

    def insert_block(self, tree=None):
        if isinstance(tree, type(None)):
            tree = self
        self.__markNodeEnds(tree)
        self.__insert_block(tree)
        results = []
        self.__findNode(tree, results)
        results = self.__cleanResult(results)
        self.results__.append(results)


def countTerminalSamples(tree):
    count = 0
    if tree.leftNode is None and tree.rightNode is None:
        count += tree.data.shape[0]
    if tree.leftNode:
        count += countTerminalSamples(tree.leftNode)
    if tree.rightNode:
        count += countTerminalSamples(tree.rightNode)
    return count

##############################################
import numpy as np

rng = np.random.RandomState(42)
X1 = np.hstack((rng.normal(-10, 2, 100), rng.normal(0, 2, 100)))
X1 = np.append(X1, -10)
X1 = np.append(X1, 0)
X2 = np.hstack((rng.normal(0, 2, 100), rng.normal(10, 2, 100)))
X2 = np.append(X2, 10)
X2 = np.append(X2, 0)
X = np.c_[X1, X2]

def plottingElements(listOfResults, exclude_list=None):
    cutoffList = [(cuts[1][0], cuts[1][1]) for cuts in listOfResults]
    cutoffList = set(cutoffList)

    return_data = {}
    for cutoff in cutoffList:
        data_list = []
        for data, (cut1, cut2) in listOfResults:
            if cutoff == (cut1, cut2):
                data_list.append(data)
        data_insert = np.vstack(data_list)
        return_data[cutoff] = data_insert

    if exclude_list:
        exclude_keys = exclude_list.keys()
        for key in exclude_keys:
            try:
                return_data.pop(key)
            except:
                pass

    return return_data

tree = Node(X, 0, 0, cut=None, block=None)

tree.insert_block()
tree.insert_block()
tree.insert_block()
tree.insert_block()

plot_data1 = plottingElements(tree.results__[0])
plot_data2 = plottingElements(tree.results__[1], plot_data1)
plot_data3 = plottingElements(tree.results__[2], plot_data2)
plot_data4 = plottingElements(tree.results__[3], plot_data3)

assert (X.shape[0] == countTerminalSamples(tree))

## 2D Plot
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], alpha=0.3, marker='o')
plt.xlabel("X[:, 0]")
plt.ylabel("X[:, 1]")

def inline_plot(plot_data, color, label):
    lw = 0.7
    for i, (cuts, data) in enumerate(plot_data.items()):
        cut_x = cuts[0]
        cut_y = cuts[1]
        plt.plot([cut_x, cut_x], [data[:, 1].min(), data[:, 1].max()], c=color, lw=lw)
        if i == 0:
            plt.plot([data[:, 0].min(), data[:, 0].max()], [cut_y, cut_y], c=color, label='depth_{}'.format(label), lw=lw)
        else:
            plt.plot([data[:, 0].min(), data[:, 0].max()], [cut_y, cut_y], c=color, lw=lw)

inline_plot(plot_data1, color='blue', label=1)
inline_plot(plot_data2, color='red', label=2)
inline_plot(plot_data3, color='green', label=3)
inline_plot(plot_data4, color='purple', label=4)

plt.legend()
plt.show()

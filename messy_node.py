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
        print(self)
        
    # def __repr__(self):
    #     return "Nozin block={} dim={} depth={} cut={}".format(self.block, self.dim, self.depth, self.cut)
        
    def insert(self, cuttoff, dim, block, cuttoffs):
        k = 5
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
            
def insert_block(tree):
    if (tree.blockEnd == True) and (tree.leftNode is None) and (tree.rightNode is None):
        block = (tree.block or 0) + 1
        print(tree.data.shape)
        cuttoffs = np.percentile(tree.data, q=50, axis=0)
        tree.insert(cuttoffs[0], 0, block=block, cuttoffs=cuttoffs)
        tree.insert(cuttoffs[1], 1, block=block, cuttoffs=cuttoffs)
        markNodeEnds(tree)
    else:
        if tree.leftNode:
            insert_block(tree.leftNode)
        if tree.rightNode:
            insert_block(tree.rightNode)
        

def getCutLevels(node, cutlist, max_level):
    cutlist.append((node.dim, node.depth, node.cut, node.block, node.min_dim1, node.min_dim2, node.max_dim1, node.max_dim2))
    if node.rightNode:
        print('right')
        getCutLevels(node.rightNode, cutlist, max_level)
    if node.leftNode:
        print('left')
        getCutLevels(node.leftNode, cutlist, max_level)

def clearCutLevels(cutlist):
    cutlist = [t for t in (list(tuple(i) for i in cutlist)) if t[2] is not None]
    cutlist.sort(key=lambda y: y[1])
    return cutlist

def markNodeEnds(tree):
    if tree.leftNode:
        currentBlock = tree.block or -1
        childBlock = tree.leftNode.block
        if (currentBlock < childBlock):
            tree.blockEnd = True
            markNodeEnds(tree.leftNode)
        elif (tree.leftNode is None) and (tree.rightNode is None):
            tree.blockEnd = True
        else:
            markNodeEnds(tree.leftNode)
            tree.blockEnd = False
    if tree.rightNode:
        currentBlock = tree.block or -1
        childBlock = tree.rightNode.block
        if (currentBlock < childBlock):
            tree.blockEnd = True
            markNodeEnds(tree.rightNode)
        elif (tree.leftNode is None) and (tree.rightNode is None):
            tree.blockEnd = True
        else:
            markNodeEnds(tree.leftNode)
            tree.blockEnd = False
    if tree.leftNode is None and tree.rightNode is None:
        tree.blockEnd = True

def countTerminalLeaves(tree):
    count = 0
    if tree.leftNode is None and tree.rightNode is None:
        count += 1
    if tree.leftNode:
        count += countTerminalLeaves(tree.leftNode)
    if tree.rightNode:
        count += countTerminalLeaves(tree.rightNode)
    return count

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
X1 = rng.normal(-100, 10, 100)
X2 = rng.normal(20, 90, 100)
X = np.c_[X1, X2]


def find_node(tree: Node, list_of_results, height=0):

    if (tree.blockEnd == True) and (tree.block == height) and (tree.data is not None):
        return (tree.data, tree.cuts)

    if tree.leftNode:
        list_of_results.append(find_node(tree.leftNode, list_of_results, height=height))
    if tree.rightNode:
        list_of_results.append(find_node(tree.rightNode, list_of_results, height=height))

def findNode(tree: Node, list_of_results):

    if (tree.leftNode is None and tree.rightNode is None):
        return (tree.data, tree.cuts)

    if tree.leftNode:
        list_of_results.append(findNode(tree.leftNode, list_of_results))
    if tree.rightNode:
        list_of_results.append(findNode(tree.rightNode, list_of_results))

def cleanResult(listOfResults):
    Result = [e for e in listOfResults if e is not None]
    return Result

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

markNodeEnds(tree)
insert_block(tree)
markNodeEnds(tree)

tree.blockEnd = False
list_of_results = []
findNode(tree, list_of_results)
list_of_results = cleanResult(list_of_results)
plot_data1 = plottingElements(list_of_results)

# Round 2
markNodeEnds(tree)
insert_block(tree)
markNodeEnds(tree)

tree.blockEnd = False
list_of_results = []
findNode(tree, list_of_results)
list_of_results = cleanResult(list_of_results)
plot_data2 = plottingElements(list_of_results, plot_data1)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])

colors = ['r', 'purple' ,'g', 'black']
for cuts, data in plot_data1.items():
    c = 'r'
    cut_x = cuts[0]
    cut_y = cuts[1]
    plt.plot([cut_x, cut_x], [data[:, 1].min(), data[:, 1].max()], c=c)
    plt.plot([data[:, 0].min(), data[:, 0].max()], [cut_y, cut_y], c=c, label='depth_1')

for cuts, data in plot_data2.items():
    c = 'g'
    cut_x = cuts[0]
    cut_y = cuts[1]
    plt.plot([cut_x, cut_x], [data[:, 1].min(), data[:, 1].max()], c=c)
    plt.plot([data[:, 0].min(), data[:, 0].max()], [cut_y, cut_y], c=c, label='depth_2')

plt.legend()
plt.show()

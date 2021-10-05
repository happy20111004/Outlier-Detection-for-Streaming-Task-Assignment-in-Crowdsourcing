import networkx as nx
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from queue import Queue

def get_weights_by_weightmatrix_and_edges(weightmatrix, edges):
    ret_weights = 0
    for node1, node2 in edges:
        ret_weights += weightmatrix[node1][node2]
    return ret_weights

def greedy_alg(weights_matrix):
    # beginning of alg

    N = weights_matrix.shape[0]

    # convert the weights_matrix to adj_dict
    # workers: [0,n_workers-1]
    # tasks: [n_workers, n_workers+n_tasks-1]
    degree_nodes = defaultdict(int)
    g = nx.Graph()
    for i in range(N):
        for j in range(N):
            if weights_matrix[i][j] != 0:
                # undirect graph
                g.add_edge(i, j+N, weight=weights_matrix[i][j])
                g.add_edge(j+N, i, weight=weights_matrix[i][j])
                degree_nodes[i] += 1
                degree_nodes[j+N] += 1

    chose_edge = []
    res = 0
    while len(g.nodes) != 0:
        # sort by degree
        node = sorted(degree_nodes.items(), key=lambda x: x[1])[0][0]
        # if there are no neighbors.
        if len(g[node]) == 0:
            g.remove_node(node)
            degree_nodes[node] = np.float("inf")
        else:
            # if the neighbor exists, we choose the maximal weight node.
            neighbor_weights = dict()
            for neighbor_node in g[node]:
                neighbor_weights[neighbor_node] = g[node][neighbor_node]["weight"]
            neighbor_node = sorted(neighbor_weights.items(), key=lambda x: x[1])[-1][0]
            res += g[node][neighbor_node]["weight"]
            if neighbor_node > node:
                chose_edge.append([node, neighbor_node-N])
            else:
                chose_edge.append([neighbor_node, node-N])
            # change the degree of neighbor nodes
            for nei_node in g[node]:
                degree_nodes[nei_node] -= 1
            for nei_node in g[neighbor_node]:
                degree_nodes[nei_node] -= 1
            # set degree of chose to INF
            degree_nodes[node] = np.float("inf")
            degree_nodes[neighbor_node] = np.float("inf")
            g.remove_nodes_from([node, neighbor_node])

    return res, chose_edge


def get_normal_worker_task_pairs(chose_edge, workers_anomaly, tasks_anomaly):
    # return the worker-task that do not contain the abnormal workers and tasks
    abnormal_workers_set = set()
    abnormal_tasks_set = set()
    for index in range(len(workers_anomaly)):
        if workers_anomaly[index] == 1:
            abnormal_workers_set.add(index)
        if tasks_anomaly[index] == 1:
            abnormal_tasks_set.add(index)
    pure_chose_edge = []
    # number of abnormal workers and abnormal tasks
    abnormal_workers, abnormal_tasks = 0, 0
    del_n_edge = 0
    for edge in chose_edge:
        if edge[0] in abnormal_workers_set or edge[1] in abnormal_tasks_set:
            del_n_edge += 1
            if edge[0] in abnormal_workers_set:
                abnormal_workers += 1
            if edge[1] in abnormal_tasks_set:
                abnormal_tasks += 1
            continue
        pure_chose_edge.append(edge)
    #print(abnormal_workers)
    #print(abnormal_tasks)
    return pure_chose_edge, len(chose_edge)-abnormal_workers, len(chose_edge)-abnormal_tasks


#vx = [0] * (N+1)
#vy = [0] * (N+1)
#px = [0] * (N+1)
#py = [0] * (N+1)
#pre = [0] * (N+1)
#lx = [0] * (N+1)
#ly = [0] * (N+1)
#slack = [0] * (N+1)
vx = None
vy = None
px = None
py = None
pre = None
lx = None
ly = None
slack = None
weights_matrix = None
N = None


def reset_KM(weights_matrix_local):
    global vx, vy, px, py, pre, lx, ly, slack, weights_matrix, N
    N = weights_matrix_local.shape[0]
    vx = [0] * (N+1)
    vy = [0] * (N+1)
    px = [0] * (N+1)
    py = [0] * (N+1)
    pre = [0] * (N+1)
    lx = [0] * (N+1)
    ly = [0] * (N+1)
    slack = [0] * (N+1)
    weights_matrix = weights_matrix_local


def aug(v: int):
    while v:
        t = px[pre[v]]
        px[pre[v]] = v
        py[v] = pre[v]
        v = t


def bfs(s: int):
    for i in range(1, N+1):
        vx[i] = 0
        vy[i] = 0
        slack[i] = np.float("inf")
    que = Queue()
    que.put(s)
    while True:
        while not que.empty():
            u = que.get()
            vx[u] = 1
            for v in range(1, N+1):
                if not vy[v]:
                    if lx[u] + ly[v] - weights_matrix[u-1][v-1] < slack[v]:
                        slack[v] = lx[u] + ly[v] - weights_matrix[u-1][v-1]
                        pre[v] = u
                        if slack[v] == 0:
                            vy[v] = 1
                            if not py[v]:
                                aug(v)
                                return
                            else:
                                que.put(py[v])
        d = np.float("inf")
        for i in range(1, N+1):
            if not vy[i]:
                d = min(d, slack[i])
        for i in range(1, N+1):
            if vx[i]:
                lx[i] -= d
            if vy[i]:
                ly[i] += d
            else:
                slack[i] -= d
        for i in range(1, N+1):
            if not vy[i]:
                if slack[i] == 0:
                    vy[i] = 1
                    if not py[i]:
                        aug(i)
                        return
                    else:
                        que.put(py[i])

def KM(weights_matrix):
    # initial KM alg
    reset_KM(weights_matrix)
    for i in range(1, N+1):
        lx[i] = max(weights_matrix[i-1])
    for i in range(1, N+1):
        bfs(i)
    ans = 0
    for i in range(1, N+1):
        ans += lx[i] + ly[i]
    #eval_ans = 0
    #for i in range(1, N+1):
    #    eval_ans += weights_matrix[i-1][px[i]-1]
    #print(eval_ans)
    return ans, px

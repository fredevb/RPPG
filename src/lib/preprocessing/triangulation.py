import numpy as np

def polygon_points_from_indexes(points, triangles):
    ret = []
    for idxs in triangles:
        ret.append([points[idx] for idx in idxs])
    return ret

def triangles_from_edges(edges, n_idx_vertices):
    adjacency_list = [[] for _ in range(0, n_idx_vertices)]
    for (v1, v2) in edges:
        adjacency_list[v1].append(v2)
        adjacency_list[v2].append(v1)
    
    res = set()
    for v1 in range(0, n_idx_vertices):
        for v2 in adjacency_list[v1]:
            for v3 in adjacency_list[v2]:
                if (v3 in adjacency_list[v1]):
                    res.add(frozenset([v1, v2, v3]))
    triangles = [list(x) for x in res]
    return triangles


def closest_point_idx(points, p):
    diff = points-p
    sq_mag = np.diagonal(diff @ diff.T)
    closest_idx = np.argmin(sq_mag)
    return closest_idx


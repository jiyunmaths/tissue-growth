"""Deterministic reference meshes for planar and closed surface simulations."""
from itertools import product
import numpy as np


def sphere_mesh(n):
    """Project an n-frequency octahedron onto the unit sphere.

    Integer lattice keys weld shared edges and poles exactly. The result has
    4*n**2+2 vertices and 8*n**2 outward-oriented, planar triangles.
    """
    points, triangles, indices = [], [], {}
    for signs in product((-1, 1), repeat=3):
        local = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                key = (signs[0]*i, signs[1]*j, signs[2]*(n-i-j))
                if key not in indices:
                    indices[key] = len(points)
                    point = np.array(key, dtype=float)
                    points.append(point / np.linalg.norm(point))
                local[i, j] = indices[key]
        for i in range(n):
            for j in range(n-i):
                triangles.append([local[i, j], local[i+1, j], local[i, j+1]])
                if i+j < n-1:
                    triangles.append([local[i+1, j], local[i+1, j+1], local[i, j+1]])
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    xyz = points[triangles]
    inward = np.einsum('ij,ij->i', np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), xyz[:, 0]) < 0
    triangles[inward] = triangles[inward][:, [0, 2, 1]]
    return points, triangles


def square_mesh(n):
    xx, yy = np.meshgrid(np.linspace(0, 1, n+1), np.linspace(0, 1, n+1))
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    i = (np.arange(n)[:, None]*(n+1)+np.arange(n)).ravel()
    triangles = np.vstack([np.column_stack([i, i+1, i+n+2]),
                           np.column_stack([i, i+n+2, i+n+1])])
    return points, triangles


def sphere_direction(x, y):
    """Normalized longitude x and colatitude y to a unit direction."""
    longitude, colatitude = 2*np.pi*x, np.pi*y
    return np.array([np.sin(colatitude)*np.cos(longitude),
                     np.sin(colatitude)*np.sin(longitude), np.cos(colatitude)])

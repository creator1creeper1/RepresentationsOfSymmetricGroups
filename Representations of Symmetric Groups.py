import sympy as sp
from frozendict import *

class Dir:
    Left = 0
    Right = 1

def compose(d, sigma, tau):
    res = {}
    for i in range(0, d):
        res[i] = sigma[tau[i]]
    return frozendict(res)

def print_permutation(d, sigma):
    if all([sigma[i] == i for i in range(d)]):
        return "e"
    cycles = cycle_decomposition(d, sigma)
    res = " ".join(["(" + " ".join([str(n + 1) for n in cycle]) + ")" for cycle in cycles if len(cycle) != 1])
    return res

def cycle_decomposition(d, sigma):
    checked = set()
    cycles = []
    while len(checked) != d:
        for i in range(d):
            if i not in checked:
                checked.add(i)
                cycle = [i]
                current_i = sigma[i]
                while current_i != i:
                    checked.add(current_i)
                    cycle.append(current_i)
                    current_i = sigma[current_i]
                cycles.append(cycle)
    return cycles

def sign(d, sigma):
    cycles = cycle_decomposition(d, sigma)
    c = len(cycles)
    return (-1)**(d - c)

def factorial(d):
    if d == 0:
        return 1
    else:
        return d * factorial(d - 1)

def get_group_algebra_matrix(d, f, g, sigma, action_dir):
    M = sp.zeros(factorial(d), factorial(d))
    for j in range(0, factorial(d)):
        if action_dir == Dir.Left:
            i = g[compose(d, sigma, f[j])]
        if action_dir == Dir.Right:
            i = g[compose(d, f[j], sigma)]
        M[i, j] = 1
    return M

def is_bijective(d, group_element):
    image = set()
    for x in range(d):
        if group_element[x] in image:
            return False
        else:
            image.add(group_element[x])
    return True

def get_group_elements(d):
    prev_templates = [frozendict()]
    for x in range(d):
        new_templates = []
        for prev_template in prev_templates:
            for y in range(d):
                new_template = dict(prev_template)
                new_template[x] = y
                new_templates.append(frozendict(new_template))
        prev_templates = new_templates.copy()
    candidates = new_templates
    group_elements = [candidate for candidate in candidates if is_bijective(d, candidate)]
    f = {i: group_elements[i] for i in range(factorial(d))}
    g = {group_elements[i]: i for i in range(factorial(d))}
    return (group_elements, f, g)

def get_group_algebra(d, group_elements, f, g, action_dir):
    matrices = dict()
    for sigma in group_elements:
        matrices[sigma] = get_group_algebra_matrix(d, f, g, sigma, action_dir)
    return matrices

def get_young_tableau(d, partition):
    i = 0
    t = []
    for j in range(len(partition)):
        row = []
        for _ in range(partition[j]):
            row.append(i)
            i += 1
        t.append(row)
    return t

def apply_to_young_tableau(d, partition, t, sigma):
    i = 0
    s = []
    for j in range(len(partition)):
        row = []
        for k in range(partition[j]):
            row.append(sigma[t[j][k]])
        s.append(row)
    return s

def same_rows(d, partition, t, s):
    for j in range(len(partition)):
        if set(t[j]) != set(s[j]):
            return False
    return True

def same_columns(d, partition, t, s):
    for i in range(d):
        t_column = set()
        s_column = set()
        for j in range(len(partition)):
            if i in range(partition[j]):
                t_column.add(t[j][i])
                s_column.add(s[j][i])
        if t_column != s_column:
            return False
    return True

def get_row_symmetrizer(d, partition, t, group_elements):
    res = []
    for sigma in group_elements:
        if same_rows(d, partition, apply_to_young_tableau(d, partition, t, sigma), t):
            res.append(sigma)
    return res

def get_column_symmetrizer(d, partition, t, group_elements):
    res = []
    for sigma in group_elements:
        if same_columns(d, partition, apply_to_young_tableau(d, partition, t, sigma), t):
            res.append(sigma)
    return res

def get_repr_matrices(d, group_elements, V_basis, V_basis_matrix, V_dim, left_matrices):
    repr_matrices = dict()
    for sigma in group_elements:
        M = sp.zeros(V_dim, V_dim)
        for j in range(V_dim):
            b = left_matrices[sigma] * V_basis[j]
            system = (V_basis_matrix, b)
            (x,) = sp.linsolve(system)
            M[:, j] = x
        repr_matrices[sigma] = M
    return repr_matrices

def get_some_repr(d, partition):
    (group_elements, f, g) = get_group_elements(d)
    left_matrices = get_group_algebra(d, group_elements, f, g, action_dir = Dir.Left)
    right_matrices = get_group_algebra(d, group_elements, f, g, action_dir = Dir.Right)
    t = get_young_tableau(d, partition)
    a = sp.zeros(factorial(d), factorial(d))
    for sigma in get_row_symmetrizer(d, partition, t, group_elements):
        a += right_matrices[sigma]
    b = sp.zeros(factorial(d), factorial(d))
    for sigma in get_column_symmetrizer(d, partition, t, group_elements):
        b += sign(d, sigma) * right_matrices[sigma]
    c = a * b
    V_basis = c.columnspace()
    V_basis_matrix = sp.Matrix.hstack(*V_basis)
    V_dim = len(V_basis)
    repr_matrices = get_repr_matrices(d, group_elements, V_basis, V_basis_matrix, V_dim, left_matrices)
    return (group_elements, V_dim, repr_matrices)

def calculate_invariant_inner_product(d, group_elements, repr_matrices, v, w):
    res = 0
    for sigma in group_elements:
        gv = repr_matrices[sigma] * v
        gw = repr_matrices[sigma] * w
        res += gv.dot(gw)
    return res

def calculate_invariant_inner_product_matrix(d, group_elements, dim, repr_matrices):
    M = sp.zeros(dim, dim)
    for j in range(dim):
        ej = sp.zeros(dim, 1)
        ej[j] = 1
        for i in range(dim):
            ei = sp.zeros(dim, 1)
            ei[i] = 1
            M[i, j] = calculate_invariant_inner_product(d, group_elements, repr_matrices, ei, ej)
    return M

def get_unitary_repr(d, partition):
    (group_elements, dim, repr_matrices) = get_some_repr(d, partition)
    inner_product_matrix = calculate_invariant_inner_product_matrix(d, group_elements, dim, repr_matrices)
    eig = inner_product_matrix.eigenvects()
    orthonormal_basis = []
    for (val, mul, vecs) in eig:
        for vec in vecs:
            vec_norm = sp.sqrt(calculate_invariant_inner_product(d, group_elements, repr_matrices, vec, vec))
            orthonormal_basis.append(vec / vec_norm)
    orthonormal_basis_matrix = sp.Matrix.hstack(*orthonormal_basis)
    unitary_matrices = dict()
    for sigma in group_elements:
        unitary_matrices[sigma] = orthonormal_basis_matrix**(-1) * repr_matrices[sigma] * orthonormal_basis_matrix
    return (group_elements, unitary_matrices)

def print_unitary_repr(d, partition):
    (group_elements, unitary_matrices) = get_unitary_repr(d, partition)
    for sigma in group_elements:
        print("{:<20} {}".format(print_permutation(d, sigma), str(unitary_matrices[sigma])))

print_unitary_repr(4, [2, 2])

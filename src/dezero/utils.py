import os
import subprocess
from pathlib import Path

from dezero.core_simple import Function, Variable


def _dot_var(v: Variable, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name)


def _dot_func(f: Function):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


def get_dot_graph(output, verbose=True):
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose=verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose=verbose)

            if x.creator is not None:
                add_func(x.creator)
    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    # Save dot language txt in a file
    save_dir = Path("./graph")
    if not save_dir.exists():
        save_dir.mkdir()
    graph_path = save_dir / "graph.dot"

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    # run dot command
    extension = os.path.splitext(to_file)[1][1:]
    cmd = f"dot {graph_path} -T {extension} -o {save_dir / to_file}"
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    import numpy as np

    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(2.0))
    x0.name = "x0"
    x1.name = "x1"
    y = x0 + x1
    x2 = Variable(np.array(2.0), name="x2")
    z = y * x2
    plot_dot_graph(z)

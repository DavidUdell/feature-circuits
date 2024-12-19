from functools import partial
from typing import Callable
from graphviz import Digraph
from collections import defaultdict
import re
import os
from tqdm import trange
import torch as t
import numpy as np
import networkx as nx

from graph_utils import iterate_edges
from config import Config
from attribution import threshold_effects
from histogram_aggregator import HistAggregator


def get_name(component, layer, idx, err_idx, add_layer=True):
    """Component is a high-level SAE location, like resid_0, or mlp_5.
    It can also be just the component name, like 'resid'.

    idx is the index of the feature within that SAE.
    err_idx is the pseudo-index of the 'error term', which we have concatenated to the end of the SAE, so
        should always be equal to d_SAE."""
    if add_layer:
        component = f"{component}_{layer}"
    match idx:
        case (seq, feat):
            if feat == err_idx:
                feat = "ε"
            return f"{seq}, {component}/{feat}"
        case (feat,):
            if feat == err_idx:
                feat = "ε"
            return f"{component}/{feat}"
        case _:
            raise ValueError(f"Invalid idx: {idx}")


def normalize_weight(weight, edge_scale, cfg: Config):
    edge_min, edge_max = edge_scale
    if cfg.edge_thickness_normalization == "linear":
        n_weight = weight / (edge_max)  # in [-1, 1]
        n_weight = (n_weight + 1) / 2  # in [0, 1]
    elif cfg.edge_thickness_normalization == "log":
        ooms = np.log(edge_max) - np.log(edge_min) + 1e-6
        n_weight = (np.log(abs(weight)) - np.log(edge_min)) / ooms  # in [0, 1]
    # rescale n_weight to be in [0.1, 1] so that minimum thickness is still visible
    scaled_weight = 0.1 + abs(n_weight) * 0.9
    return str(scaled_weight * cfg.pen_thickness)


def _to_hex(number, scale):
    number = number / scale

    # Define how the intensity changes based on the number
    # - Negative numbers increase red component to max
    # - Positive numbers increase blue component to max
    # - 0 results in white
    if number < 0:
        # Increase towards red, full intensity at -1.0
        red = 255
        green = blue = int(
            (1 + number) * 255
        )  # Increase other components less as it gets more negative
    elif number > 0:
        # Increase towards blue, full intensity at 1.0
        blue = 255
        red = green = int(
            (1 - number) * 255
        )  # Increase other components less as it gets more positive
    else:
        # Exact 0, resulting in white
        red = green = blue = 255

    # decide whether text is black or white depending on darkness of color
    text_hex = (
        "#000000"
        if (red * 0.299 + green * 0.587 + blue * 0.114) > 170
        else "#ffffff"
    )

    # Convert to hex, ensuring each component is 2 digits
    hex_code = f"#{red:02X}{green:02X}{blue:02X}"

    return hex_code, text_hex


def _get_label(name, annotations=None):
    if annotations is None:
        return name
    else:
        match name.split(", "):
            case seq, feat:
                if feat in annotations:
                    component = feat.split("/")[0]
                    component = feat.split("_")[0]
                    return f"{seq}, {annotations[feat]} ({component})"
                return name
            case [feat]:
                if feat in annotations:
                    component = feat.split("/")[0]
                    component = feat.split("_")[0]
                    return f"{annotations[feat]} ({component})"


def nodes_by_submod_add_entry(
    nodes, nodes_by_submod, node_name, cfg: Config, hist_agg: HistAggregator
):
    """Add a node to the submodule index/effect dict."""

    submod_nodes = nodes[node_name].to_tensor()
    topk_ind = threshold_effects(submod_nodes, cfg, node_name, hist_agg)

    nodes_by_submod[node_name] = {
        tuple(idx): submod_nodes[tuple(idx)].item() for idx in topk_ind
    }


def plot_circuit(
    nodes,
    edges,
    annotations,
    cfg: Config,
    hist_agg: HistAggregator,
    example_text: str | None = None,
    save_path: str = "",
):
    """Plot and render the overall circuit."""

    if cfg.aggregation == "none":
        plot_circuit_posaligned(
            nodes, edges, annotations, example_text, cfg, save_path
        )
        return

    # get min and max node effects
    mins = {n: v.to_tensor().min() for n, v in nodes.items() if n != "y"}
    maxs = {n: v.to_tensor().max() for n, v in nodes.items() if n != "y"}
    min_effect = min(mins.values())
    max_effect = max(maxs.values())
    scale = max(abs(min_effect), abs(max_effect))
    to_hex = partial(_to_hex, scale=scale)
    get_label = partial(_get_label, annotations=annotations)

    # The fleshing out of nodes_by_submod is what to watch here, for
    # replicability.
    nodes_by_submod = {}

    if cfg.first_component == "embed":
        nodes_by_submod_add_entry(
            nodes, nodes_by_submod, "embed", cfg, hist_agg
        )

    for layer in range(cfg.layers):
        for component in ["attn", "mlp", "resid"]:
            node_name = f"{component}_{layer}"
            nodes_by_submod_add_entry(
                nodes, nodes_by_submod, node_name, cfg, hist_agg
            )

    pruned_graph = build_pruned_graph(nodes, edges, nodes_by_submod, cfg)
    graph = build_formatted_graph(
        nodes,
        edges,
        to_hex,
        get_label,
        cfg,
        nodes_by_submod,
        example_text,
        pruned_graph,
    )

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    graph.render(save_path, format="png", cleanup=False)


def compute_edge_scale(jacobian):
    if jacobian.values().shape[0] == 0:
        return None, None
    # smallest non-zero abs(edge weight)
    edge_min = float(abs(jacobian.values()[jacobian.values().nonzero()]).min())
    if edge_min == 0.0:
        pass
    # largest edge weight
    edge_max = float(
        max(abs(jacobian.values().min()), abs(jacobian.values().max()))
    )
    return edge_min, edge_max


def iterate_edge_feats(nodes, edges, nodes_by_submod, layer, cfg: Config):
    for up, down in iterate_edges(edges, layer, cfg.first_component):
        jacobian = edges[up][down]
        err_idx = nodes[up].shape[0]
        edge_scale = compute_edge_scale(jacobian)
        if jacobian.shape[0] == 0:
            continue
        for upstream_idx in nodes_by_submod[up].keys():
            for downstream_idx in nodes_by_submod[down].keys():
                weight = jacobian[tuple(downstream_idx)][
                    tuple(upstream_idx)
                ].item()
                uname = get_name(
                    up, layer, upstream_idx, err_idx, add_layer=False
                )
                dname = get_name(
                    down, layer, downstream_idx, err_idx, add_layer=False
                )
                if abs(weight) > 0:
                    yield uname, dname, weight, edge_scale


def build_pruned_graph(nodes, edges, nodes_by_submod, cfg: Config):
    if cfg.prune_method == "none":  # do no pruning
        return None

    G = nx.DiGraph()

    # construct graph in networkx
    for layer in trange(cfg.layers):
        for uname, dname, _, _ in iterate_edge_feats(
            nodes, edges, nodes_by_submod, layer, cfg
        ):
            G.add_edge(uname, dname)

    # add final edge to output node 'y'
    if cfg.resid_posn == "post":
        err_idx = nodes[f"resid_{cfg.layers-1}"].shape[0]
        terminal_node = "y"
        for idx in nodes_by_submod[f"resid_{cfg.layers-1}"].keys():
            name = get_name("resid", cfg.layers - 1, idx, err_idx)
            G.add_edge(name, terminal_node)

    # restrict to nodes that have a path to terminal
    reversed_graph = G.reverse(copy=False)
    nodes_reaching_terminal = nx.descendants(reversed_graph, terminal_node)
    nodes_reaching_terminal.add(terminal_node)

    if cfg.prune_method in ["source-sink", "first-layer-sink"]:
        # restrict to nodes that are reachable from a starter node
        relevant_nodes = set()
        if cfg.prune_method == "first-layer-sink":
            if cfg.first_component == "embed":
                starter_components = ["embed"]
            else:
                starter_components = ["attn_0", "mlp_0", "resid_0"]
        else:
            starter_components = [cfg.first_component]

        for (
            component
        ) in (
            starter_components
        ):  # allow any first layer node type to be a starter
            starter_nodes = nodes_by_submod[
                component
            ].keys()  # check all features in this component
            for starter in starter_nodes:
                err_idx = nodes[component].shape[0]
                s = get_name(
                    component, 0, starter, err_idx, add_layer=False
                )  # if this feature reaches the end, then add all descendant nodes
                if (
                    s in nodes_reaching_terminal
                ):  # if the starter actually goes to terminal
                    reachable_nodes = nx.descendants(
                        G, s
                    )  # all nodes reachable from this starter node
                    relevant_nodes.update(
                        reachable_nodes.intersection(nodes_reaching_terminal)
                    )  # if go to terminal and reachable by this starter
                    relevant_nodes.add(s)

    elif cfg.prune_method == "sink-backwards":
        # restrict to nodes that reach the output, regardless of where they start
        relevant_nodes = nodes_reaching_terminal

    # subgraph with only nodes that are on a path from resid_{start} to y
    pruned_graph = G.subgraph(relevant_nodes).copy()
    return pruned_graph


def add_node_prune_filter(name, graph, pruned_graph, **kwargs):
    if pruned_graph is not None and name not in pruned_graph.nodes:
        return
    graph.node(name, **kwargs)


def add_edge_prune_filter(uname, dname, graph, pruned_graph, **kwargs):
    if pruned_graph is not None and (uname, dname) not in pruned_graph.edges:
        return
    graph.edge(uname, dname, **kwargs)


def add_layer_nodes_to_graph(
    graph: Digraph,
    component: str,
    layer: int,
    nodes: dict[str, t.Tensor],
    nodes_by_submod: dict[str, t.Tensor],
    to_hex: Callable,
    get_label: Callable,
    pruned_graph: nx.DiGraph,
):
    """Add layer nodes to a DiGraph object."""

    component_name = f"{component}_{layer}" if layer != -1 else component
    err_idx = nodes[component_name].shape[0]
    with graph.subgraph(name=f"layer {layer} {component}") as subgraph:
        subgraph.attr(rank="same")
        max_seq_pos = None
        for idx, effect in nodes_by_submod[component_name].items():
            name = get_name(component, layer, idx, err_idx)
            fillhex, texthex = to_hex(effect)

            if name[-1:].endswith("ε"):
                add_node_prune_filter(
                    name,
                    subgraph,
                    pruned_graph,
                    shape="triangle",
                    width="1.6",
                    height="0.8",
                    fixedsize="true",
                    fillcolor=fillhex,
                    style="filled",
                    fontcolor=texthex,
                )
            else:
                add_node_prune_filter(
                    name,
                    subgraph,
                    pruned_graph,
                    label=get_label(name),
                    fillcolor=fillhex,
                    fontcolor=texthex,
                    style="filled",
                )
            # if sequence position is present, separate nodes by sequence
            # position
            match idx:
                case (seq, _):
                    subgraph.node(
                        f"{component}_{layer}_#{seq}_pre", style="invis"
                    )
                    subgraph.node(
                        f"{component}_{layer}_#{seq}_post", style="invis"
                    )

                    subgraph.edge(
                        f"{component}_{layer}_#{seq}_pre", name, style="invis"
                    )
                    subgraph.edge(
                        name, f"{component}_{layer}_#{seq}_post", style="invis"
                    )

                    if max_seq_pos is None or seq > max_seq_pos:
                        max_seq_pos = seq

        if max_seq_pos is None:
            return
        # make sure the auxiliary ordering nodes are in right order
        for seq in reversed(range(max_seq_pos + 1)):
            if f"{component}_{layer}_#{seq}_pre" in "".join(subgraph.body):
                for seq_prev in range(seq):
                    if f"{component}_{layer}_#{seq_prev}_post" in "".join(
                        subgraph.body
                    ):
                        subgraph.edge(
                            f"{component}_{layer}_#{seq_prev}_post",
                            f"{component}_{layer}_#{seq}_pre",
                            style="invis",
                        )


def build_formatted_graph(
    nodes,
    edges,
    to_hex,
    get_label,
    cfg: Config,
    nodes_by_submod,
    example_text,
    pruned_graph,
):
    """Build out the formatted Digraph object."""

    graph = Digraph(name="Feature circuit")

    graph.graph_attr.update(rankdir="BT", newrank="true")
    graph.node_attr.update(shape="box", style="rounded")

    if cfg.first_component == "embed":
        add_layer_nodes_to_graph(
            graph,
            "embed",
            -1,
            nodes,
            nodes_by_submod,
            to_hex,
            get_label,
            pruned_graph,
        )

    for layer in trange(cfg.layers):
        # add all nodes for this layer
        for component in ["attn", "mlp", "resid"]:
            add_layer_nodes_to_graph(
                graph,
                component,
                layer,
                nodes,
                nodes_by_submod,
                to_hex,
                get_label,
                pruned_graph,
            )

        # add all edges
        for uname, dname, weight, edge_scale in iterate_edge_feats(
            nodes, edges, nodes_by_submod, layer, cfg
        ):
            add_edge_prune_filter(
                uname,
                dname,
                graph,
                pruned_graph,
                penwidth=normalize_weight(weight, edge_scale, cfg),
                color="red" if weight < 0 else "blue",
            )

    formatted_text = example_text.replace("\\", "\\\\")
    if cfg.resid_posn == "post":
        add_node_prune_filter(
            "y", graph, pruned_graph, shape="diamond", xlabel=formatted_text
        )
        err_idx = nodes[f"resid_{cfg.layers-1}"].shape[0]
        jacobian = edges[f"resid_{cfg.layers-1}"]["y"]
        edge_scale = compute_edge_scale(jacobian)

        for idx in nodes_by_submod[f"resid_{cfg.layers-1}"].keys():
            weight = jacobian[tuple(idx)].item()
            name = get_name("resid", cfg.layers - 1, idx, err_idx)

            add_edge_prune_filter(
                name,
                "y",
                graph,
                pruned_graph,
                penwidth=normalize_weight(weight, edge_scale, cfg),
                color="red" if weight < 0 else "blue",
            )

    return graph


def plot_circuit_posaligned(
    nodes, edges, annotations, example_text, cfg: Config, save_dir: str
):
    raise NotImplementedError
    # # get min and max node effects
    # min_effect = min([v.to_tensor().min() for n, v in nodes.items() if n != 'y'])
    # max_effect = max([v.to_tensor().max() for n, v in nodes.items() if n != 'y'])
    # scale = max(abs(min_effect), abs(max_effect))
    # to_hex = partial(_to_hex, scale=scale)
    # get_label = partial(_get_label, annotations=annotations)

    # words = example_text.split()

    # G = Digraph(name='Feature circuit')
    # G.graph_attr.update(rankdir='BT', newrank='true')
    # G.node_attr.update(shape="box", style="rounded")

    # start_layer = 0
    # if 'embed' in nodes:
    #     nodes_by_submod = {
    #         'resid_-1' : {tuple(idx.tolist()) : nodes['embed'].to_tensor()[tuple(idx)].item() for idx in (nodes['embed'].to_tensor().abs() > cfg.node_sparsity).nonzero()}
    #     }
    #     edges['resid_-1'] = edges['embed']
    #     start_layer = -1
    # nodes_by_seqpos = defaultdict(list)
    # nodes_by_layer = defaultdict(list)
    # edgeset = set()

    # for layer in range(cfg.layers):
    #     for component in ['attn', 'mlp', 'resid']:
    #         submod_nodes = nodes[f'{component}_{layer}'].to_tensor()
    #         nodes_by_submod[f'{component}_{layer}'] = {
    #             tuple(idx.tolist()) : submod_nodes[tuple(idx)].item() for idx in (submod_nodes.abs() > cfg.node_sparsity).nonzero()
    #         }

    # # add words to bottom of graph
    # with G.subgraph(name=f'words') as subgraph:
    #     subgraph.attr(rank='same')
    #     prev_word = None
    #     for idx in range(cfg.example_length):
    #         word = words[idx]
    #         subgraph.node(word, shape='none', group=str(idx), fillcolor='transparent',
    #                       fontsize="30pt")
    #         if prev_word is not None:
    #             subgraph.edge(prev_word, word, style='invis', minlen="2")
    #         prev_word = word

    # for layer in range(start_layer, cfg.layers):
    #     for component in ['attn', 'mlp', 'resid']:
    #         if layer == start_layer and component != 'resid': continue
    #         with G.subgraph(name=f'layer {layer} {component}') as subgraph:
    #             subgraph.attr(rank='same')
    #             max_seq_pos = None
    #             for idx, effect in nodes_by_submod[f'{component}_{layer}'].items():
    #                 name = get_name(component, layer, idx)
    #                 seq_pos, basename = name.split(", ")
    #                 fillhex, texthex = to_hex(effect)
    #                 if name[-1:] == 'ε':
    #                     subgraph.node(name, shape='triangle', group=seq_pos, width="1.6", height="0.8", fixedsize="true",
    #                                   fillcolor=fillhex, style='filled', fontcolor=texthex)
    #                 else:
    #                     subgraph.node(name, label=get_label(name), group=seq_pos, fillcolor=fillhex, fontcolor=texthex,
    #                                   style='filled')

    #                 if len(nodes_by_seqpos[seq_pos]) == 0:
    #                     G.edge(words[int(seq_pos)], name, style='dotted', arrowhead='none', penwidth="1.5")
    #                     edgeset.add((words[int(seq_pos)], name))

    #                 nodes_by_seqpos[seq_pos].append(name)
    #                 nodes_by_layer[layer].append(name)

    #                 # if sequence position is present, separate nodes by sequence position
    #                 match idx:
    #                     case (seq, _):
    #                         subgraph.node(f'{component}_{layer}_#{seq}_pre', style='invis'), subgraph.node(f'{component}_{layer}_#{seq}_post', style='invis')
    #                         subgraph.edge(f'{component}_{layer}_#{seq}_pre', name, style='invis'), subgraph.edge(name, f'{component}_{layer}_#{seq}_post', style='invis')
    #                         if max_seq_pos is None or seq > max_seq_pos:
    #                             max_seq_pos = seq

    #             if max_seq_pos is None: continue
    #             # make sure the auxiliary ordering nodes are in right order
    #             for seq in reversed(range(max_seq_pos+1)):
    #                 if f'{component}_{layer}_#{seq}_pre' in ''.join(subgraph.body):
    #                     for seq_prev in range(seq):
    #                         if f'{component}_{layer}_#{seq_prev}_post' in ''.join(subgraph.body):
    #                             subgraph.edge(f'{component}_{layer}_#{seq_prev}_post', f'{component}_{layer}_#{seq}_pre', style='invis')

    #     for component in ['attn', 'mlp']:
    #         if layer == -1: continue
    #         for upstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
    #             for downstream_idx in nodes_by_submod[f'resid_{layer}'].keys():
    #                 weight = edges[f'{component}_{layer}'][f'resid_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
    #                 if abs(weight) > edge_sparsity:
    #                     uname = get_name(component, layer, upstream_idx)
    #                     dname = get_name('resid', layer, downstream_idx)
    #                     G.edge(
    #                         uname, dname,
    #                         penwidth=str(abs(weight) * pen_thickness),
    #                         color = 'red' if weight < 0 else 'blue'
    #                     )
    #                     edgeset.add((uname, dname))

    #     # add edges to previous layer resid
    #     for component in ['attn', 'mlp', 'resid']:
    #         if layer == -1: continue
    #         for upstream_idx in nodes_by_submod[f'resid_{layer-1}'].keys():
    #             for downstream_idx in nodes_by_submod[f'{component}_{layer}'].keys():
    #                 weight = edges[f'resid_{layer-1}'][f'{component}_{layer}'][tuple(downstream_idx)][tuple(upstream_idx)].item()
    #                 if abs(weight) > edge_sparsity:
    #                     uname = get_name('resid', layer-1, upstream_idx)
    #                     dname = get_name(component, layer, downstream_idx)
    #                     G.edge(
    #                         uname, dname,
    #                         penwidth=str(abs(weight) * pen_thickness),
    #                         color = 'red' if weight < 0 else 'blue'
    #                     )
    #                     edgeset.add((uname, dname))

    # # the cherry on top
    # G.node('y', shape='diamond')
    # for idx in nodes_by_submod[f'resid_{layers-1}'].keys():
    #     weight = edges[f'resid_{layers-1}']['y'][tuple(idx)].item()
    #     if abs(weight) > edge_sparsity:
    #         name = get_name('resid', layers-1, idx)
    #         G.edge(
    #             name, 'y',
    #             penwidth=str(abs(weight) * pen_thickness),
    #             color = 'red' if weight < 0 else 'blue'
    #         )
    #         edgeset.add((uname, dname))

    # if not os.path.exists(os.path.dirname(save_dir)):
    #     os.makedirs(os.path.dirname(save_dir))
    # G.render(save_dir, format='png', cleanup=True)

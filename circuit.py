"""Run this script directly with `python3 circuit.py` for plotting."""

import argparse
import gc
import json
import math
import os
from collections import defaultdict
from textwrap import dedent
import random

import torch as t
from tqdm import tqdm


from histogram_aggregator import HistAggregator, ThresholdType
from attribution import patching_effect, jvp, threshold_effects
from circuit_plotting import plot_circuit
from load_model import load_hists, load_model_dicts
from loading_utils import get_examples
from config import Config
from tensor_ops import sparse_select_last


def aggregate_single_node_edge(
    is_node, weight_matrix, dims, divisor, last_agg
):
    if last_agg:
        if is_node:
            weight_matrix = sparse_select_last(weight_matrix, dims[0])
        else:
            weight_matrix = sparse_select_last(weight_matrix, dims)
    else:
        if is_node:
            weight_matrix = weight_matrix.sum(dim=dims[0]) / divisor
        else:
            weight_matrix = weight_matrix.sum(dim=dims) / divisor
    return weight_matrix


def aggregate_nodes_edges(
    nodes,
    edges,
    dims: tuple[int],
    divisor_dim: int = None,
    last_agg: bool = False,
):
    """Apply aggregation method to all nodes and edges.

    Args:
        nodes (dict): dictionary of nodes
        edges (dict): dictionary of edges
        dims (tuple): dimensions to aggregate over
        divisor_dim (int): if last_agg=False, then results will be divided by
        the size of this dimension. Otherwise has no effect.
        last_agg (bool): if True, then the last element of the specified
        dimension for each dimension in dims will be selected
    """

    for child in edges:
        for parent in edges[child]:

            if divisor_dim is not None:
                divisor = nodes[child].shape[divisor_dim]
            else:
                divisor = 1.0

            weight_matrix = edges[child][parent]
            if weight_matrix.shape[0] == 0:
                continue

            if parent == "y":
                weight_matrix = aggregate_single_node_edge(
                    True, weight_matrix, dims, divisor, last_agg
                )
            else:
                weight_matrix = aggregate_single_node_edge(
                    False, weight_matrix, dims, divisor, last_agg
                )

            edges[child][parent] = weight_matrix

    for node in nodes:
        if node != "y":
            slices = [slice(None) for _ in range(len(nodes[node].shape))]
            slices[dims[0]] = -1
            n = nodes[node]
            n.act = n.act[slices]
            n.resc = n.resc[slices]


def clear_cache():
    """Simply clear cuda and Python memory."""
    t.cuda.empty_cache()
    gc.collect()


def get_circuit(
    clean,
    patch,
    model,
    embed,
    attns,
    mlps,
    resids,
    dictionaries,
    cfg: Config,
    hist_agg: HistAggregator,
    metric_fn,
    metric_kwargs={},
):

    all_submods = ([embed] if embed is not None else []) + [
        submod
        for layer_submods in zip(mlps, attns, resids)
        for submod in layer_submods
    ]

    # first get the patching effect of everything on y
    # Key intermediate values exposed in here.
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method=cfg.method,  # ig gets better approximations for early layers
    )
    # effects: dict[nnsight.envoy.Envoy]
    # deltas: dict[nnsight.envoy.Envoy]
    # grads: dict[nnsight.envoy.Envoy]
    # total_effect: None

    clear_cache()  # helps a bit with memory management

    features_by_submod = {}

    # print("Effect Sums:")
    for submod in all_submods:
        effect = effects[submod].to_tensor()
        # effect_last = effect[:, -1, :]

        # # Debug program state print
        # print(
        #     get_submod_repr(submod),
        #     str(list(effect_last.shape)) + ":",
        #     round(effect_last.sum().item(), 1),
        # )

        if cfg.collect_hists > 0:
            hist_agg.compute_node_hist(submod, effect)
        features_by_submod[submod] = threshold_effects(
            effect, cfg, submod, hist_agg
        )
        # print(
        #     "\tn_feats",
        #     get_submod_repr(submod),
        #     len(features_by_submod[submod]),
        # )
    # print()

    if len(features_by_submod[resids[-1]]) == 0:
        print("No features found for last layer. Skipping...")
        return None, None

    # submodule -> list of indices

    n_layers = len(resids)

    nodes = {"y": total_effect}
    if embed is not None:
        nodes["embed"] = effects[embed]
    for i in range(n_layers):
        nodes[f"attn_{i}"] = effects[attns[i]]
        nodes[f"mlp_{i}"] = effects[mlps[i]]
        nodes[f"resid_{i}"] = effects[resids[i]]

    if cfg.nodes_only:
        if cfg.aggregation == "sum":
            for k in nodes:
                if k != "y":
                    nodes[k] = nodes[k].sum(dim=1)
        if cfg.aggregation == "last":
            for k in nodes:
                if k != "y":
                    nodes[k] = nodes[k][:, -1]
        nodes = {k: v.mean(dim=0) for k, v in nodes.items()}
        return nodes, None

    edges = defaultdict(lambda: {})
    edges[f"resid_{len(resids)-1}"] = {
        "y": effects[resids[-1]].to_tensor().to_sparse()
    }

    def N(upstream, downstream, intermediate_stop_grads=None):
        return jvp(
            clean,
            model,
            dictionaries,
            downstream,  # downstream submod
            features_by_submod[downstream],  # downstream features
            upstream,  # upstream submod
            grads[downstream],  # left_vec
            deltas[upstream],  # right_vec
            cfg,
            hist_agg,
            intermediate_stop_grads=intermediate_stop_grads,
        )

    # now we work backward through the model to get the edges
    for layer in reversed(range(len(resids))):
        resid = resids[layer]
        mlp = mlps[layer]
        attn = attns[layer]

        MR_effect = N(mlp, resid)
        AR_effect = N(attn, resid, [mlp])

        edges[f"mlp_{layer}"][f"resid_{layer}"] = MR_effect
        edges[f"attn_{layer}"][f"resid_{layer}"] = AR_effect

        if not cfg.parallel_attn:
            AM_effect = N(attn, mlp)
            edges[f"attn_{layer}"][f"mlp_{layer}"] = AM_effect

        if layer > 0:
            prev_resid = resids[layer - 1]
            upstream_name = f"resid_{layer-1}"
        else:
            if embed is not None:
                prev_resid = embed
                upstream_name = "embed"
            else:
                continue

        RM_effect = N(prev_resid, mlp, [attn])
        RA_effect = N(prev_resid, attn)
        RR_effect = N(prev_resid, resid, [mlp, attn])

        edges[upstream_name][f"mlp_{layer}"] = RM_effect
        edges[upstream_name][f"attn_{layer}"] = RA_effect
        edges[upstream_name][f"resid_{layer}"] = RR_effect
        print("layer done", layer)
        clear_cache()

    if cfg.aggregation == "sum":
        # aggregate across sequence position
        aggregate_nodes_edges(nodes, edges, (1, 4))
        # aggregate across batch dimension
        aggregate_nodes_edges(nodes, edges, (0, 2), divisor_dim=0)

    elif cfg.aggregation == "none":
        # aggregate across batch dimensions
        aggregate_nodes_edges(nodes, edges, (0, 3), divisor_dim=0)
    elif cfg.aggregation == "last":
        # select last sequence position
        aggregate_nodes_edges(
            nodes, edges, (1, 4), divisor_dim=0, last_agg=True
        )
        # sum aggregate across batch dim
        aggregate_nodes_edges(nodes, edges, (0, 2), divisor_dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {cfg.aggregation}")

    return nodes, edges


def process_examples(
    model,
    embed,
    attns,
    mlps,
    resids,
    dictionaries,
    example_basename,
    examples,
    cfg: Config,
    hist_agg: HistAggregator,
):
    # Suppress transformers deprecation warning
    model.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = (
        True
    )
    batch_size = cfg.batch_size
    num_examples = min([cfg.num_examples, len(examples)])
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]
    if (
        num_examples < cfg.num_examples and not cfg.disable_tqdm
    ):  # warn the user
        print(
            dedent(
                f"""
                Total number of examples is less than {cfg.num_examples}. Using
                {num_examples} examples instead.
                """
            )
        )

    if not cfg.plot_only:
        nodes, edges = compute_circuit(
            model,
            embed,
            attns,
            mlps,
            resids,
            dictionaries,
            example_basename,
            examples,
            cfg,
            hist_agg,
            num_examples,
            batches,
        )
        if nodes is None and edges is None:
            return
    else:
        with open(
            f"{cfg.circuit_dir}/{example_basename}_{cfg.as_fname()}.pt", "rb"
        ) as infile:
            save_dict = t.load(infile)
        nodes = save_dict["nodes"]
        edges = save_dict["edges"]

    # feature annotations
    try:
        annotations = {}
        with open(
            f"annotations/{cfg.dict_id}.jsonl", "r", encoding="utf-8"
        ) as annotations_data:
            for annotation_line in annotations_data:
                annotation = json.loads(annotation_line)
                annotations[annotation["Name"]] = annotation["Annotation"]
    except FileNotFoundError:
        annotations = None

    example_text = None
    if cfg.data_type in ["hf", "prompt"]:
        example_text = (
            "".join(model.tokenizer.decode(examples[0]["clean_prefix"][0]))
            + " -> "
            + model.tokenizer.decode(examples[0]["clean_answer"])
        )
    elif cfg.aggregation == "none":
        example_text = model.tokenizer.batch_decode(
            examples[0]["clean_prefix"]
        )[0]

    if cfg.collect_hists == 0:
        plot_circuit(
            nodes,
            edges,
            annotations,
            cfg,
            hist_agg,
            example_text,
            save_path=f"{cfg.plot_dir}/{example_basename}_{cfg.as_fname()}",
        )


def compute_circuit(
    model,
    embed,
    attns,
    mlps,
    resids,
    dictionaries,
    example_basename,
    examples,
    cfg: Config,
    hist_agg,
    num_examples,
    batches,
):
    running_nodes = None
    running_edges = None

    for batch in tqdm(batches, desc="Batches", disable=cfg.disable_tqdm):
        clean_inputs = t.cat([e["clean_prefix"] for e in batch], dim=0).to(
            cfg.device
        )
        clean_answer_idxs = (
            t.tensor(
                [e["clean_answer"] for e in batch],
                dtype=t.long,
                device=cfg.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        print("Index tensor:", clean_answer_idxs, clean_answer_idxs.shape)
        if cfg.model == "gpt2":
            model_out = model.lm_head
        else:
            model_out = model.embed_out

        if cfg.data_type in ["nopair", "hf", "prompt"]:
            patch_inputs = None

            def metric_fn(model):
                metric = t.nn.CrossEntropyLoss()
                # clean_answer_idxs: tensor(220, device='cuda:0')
                logits = model_out.output[0, 0, -1, :].save()
                loss = metric(
                    logits,
                    clean_answer_idxs.squeeze(),
                )
                return loss, logits
                # return -1 * t.gather(
                #     t.nn.functional.log_softmax(
                #         model_out.output[:, -1, :], dim=-1
                #     ),
                #     dim=-1,
                #     index=clean_answer_idxs,
                # ).squeeze(-1)

        else:
            raise ValueError("Should not have been reached.")
            # patch_inputs = t.cat([e["patch_prefix"] for e in batch],
            #     dim=0).to( cfg.device
            # )
            # patch_answer_idxs = t.tensor(
            #     [e["patch_answer"] for e in batch],
            #     dtype=t.long,
            #     device=cfg.device,
            # )

            # def metric_fn(model):
            #     return t.gather(
            #         model_out.output[:, -1, :],
            #         dim=-1,
            #         index=patch_answer_idxs.view(-1, 1),
            #     ).squeeze(-1) - t.gather(
            #         model_out.output[:, -1, :],
            #         dim=-1,
            #         index=clean_answer_idxs.view(-1, 1),
            #     ).squeeze(
            #         -1
            #     )

        nodes, edges = get_circuit(
            clean_inputs,
            patch_inputs,
            model,
            embed,
            attns,
            mlps,
            resids,
            dictionaries,
            cfg,
            hist_agg,
            metric_fn,
            metric_kwargs=dict(),
        )

        if nodes is None and edges is None:
            return None, None

        running_nodes = {
            k: len(batch) * nodes[k].to("cpu")
            for k in nodes.keys()
            if k != "y"
        }
        if not cfg.nodes_only:
            running_edges = {
                k: {
                    kk: len(batch) * edges[k][kk].to("cpu")
                    for kk in edges[k].keys()
                }
                for k in edges.keys()
            }

            # memory cleanup
        del nodes, edges
        gc.collect()

    nodes = {
        k: v.to(cfg.device) / num_examples for k, v in running_nodes.items()
    }
    if not cfg.nodes_only:
        edges = {
            k: {
                kk: 1 / num_examples * v.to(cfg.device)
                for kk, v in running_edges[k].items()
            }
            for k in running_edges.keys()
        }
    else:
        edges = None

    save_dict = {"examples": examples, "nodes": nodes, "edges": edges}
    if cfg.collect_hists == 0:
        with open(
            f"{cfg.circuit_dir}/{example_basename}_{cfg.as_fname()}.pt", "wb"
        ) as outfile:
            t.save(save_dict, outfile)

    return nodes, edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="simple_train",
        help=dedent(
            """
            A subject-verb agreement dataset in data/, or a path to a 
            cluster.json.
            """
        ),
    )
    parser.add_argument(
        "--num_examples",
        "-n",
        type=int,
        default=100,
        help=dedent(
            """
            The number of examples from the --dataset over which to average
            indirect effects.
            """
        ),
    )
    parser.add_argument(
        "--example_length",
        "-l",
        type=int,
        default=None,
        help=dedent(
            """
            The max length (if using sum aggregation) or exact length (if not
            aggregating) of examples.
            """
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="The Huggingface ID of the model you wish to test.",
    )
    parser.add_argument(
        "--dict_path",
        type=str,
        default="dictionaries/pythia-70m-deduped/",
        help="Path to all dictionaries for your language model.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Hidden size of the language model.",
    )
    parser.add_argument(
        "--dict_id",
        type=str,
        default="gpt2",
        help=dedent(
            """
            ID of the dictionaries. Use `id` to obtain circuits on
            neurons/heads directly.
            """
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=dedent(
            """
            Number of examples to process at once when running circuit
            discovery.
            """
        ),
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="sum",
        choices=["sum", "last", "none"],
        help=dedent(
            """
            Aggregation across token positions. Should be one of `sum` or
            `none`.
            """
        ),
    )
    parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.2,
        help="Indirect effect threshold for keeping circuit nodes.",
    )
    parser.add_argument(
        "--node_thresh_type",
        type=ThresholdType,
        default=ThresholdType.THRESH,
        choices=list(ThresholdType),
        help="Threshold type for node_threshold.",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=50,
        help=dedent(
            """
            Limit feats/submod to at most this many, regardless of thresholding
            strategy (topk that pass threshold test)
            """
        ),
    )
    parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.02,
        help="Indirect effect threshold for keeping edges.",
    )
    parser.add_argument(
        "--edge_thresh_type",
        type=ThresholdType,
        default=ThresholdType.THRESH,
        choices=list(ThresholdType),
        help="Threshold type for edge_threshold.",
    )
    parser.add_argument(
        "--pen_thickness",
        type=float,
        default=0.5,
        help="Scales the width of the edges in the circuit plot.",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        default="none",
        choices=["none", "source-sink", "sink-backwards", "first-layer-sink"],
        help=dedent(
            """
            Pruning method for the finished circuit (see
            circuit_plotting.build_pruned_graph)
            """
        ),
    )
    parser.add_argument(
        "--edge_thickness_normalization",
        type=str,
        choices=["log", "linear"],
        default="log",
        help="Specify the normalization type for edge thickness.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["nopair", "regular", "hf", "prompt"],
        default="prompt",
        help="Specify the type of the dataset.",
    )
    # The repo currently seems to chop off the last token in the sequence.
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, world",
        help=dedent(
            """
            Input a custom prompt to generate a circuit on. Only used when
            data_type is 'prompt'.
            """
        ),
    )

    hist_options = parser.add_argument_group("Histogram collection options")
    hist_options.add_argument(
        "--collect_hists",
        default=0,
        type=int,
        help=dedent(
            """
            Collect histograms of edge and node weights for the first
            collect_hists examples rather than compute circuits. 0 to disable.
            """
        ),
    )
    hist_options.add_argument(
        "--accumulate_hists",
        default=False,
        action="store_true",
        help=dedent(
            """
            Accumulate histograms from existing files in the circuit directory.
            """
        ),
    )
    hist_options.add_argument(
        "--histogram_path",
        type=str,
        default="",
        help="Path to histograms for thresholding.",
    )
    hist_options.add_argument(
        "--bootstrap_path",
        default=None,
        type=str,
        help=dedent(
            """
            If set, histogram_path will be used to load an existing histogram
            to set thresholds, and then a new one will be written to at
            bootstrap_path.
            """
        ),
    )

    parser.add_argument(
        "--plot_circuit",
        default=False,
        action="store_true",
        help="Plot the circuit after discovering it.",
    )
    parser.add_argument(
        "--nodes_only",
        default=False,
        action="store_true",
        help=dedent(
            """
            Only search for causally implicated features; do not draw edges.
            """
        ),
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Do not run circuit discovery; just plot an existing circuit.",
    )
    parser.add_argument(
        "--circuit_dir",
        type=str,
        default="circuits/",
        help="Directory to save/load circuits.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="circuits/figures/",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling examples.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    cfg = Config()
    cfg.update(args)

    model, embed, attns, mlps, resids, dictionaries = load_model_dicts(
        args, cfg
    )

    save_basename, examples = get_examples(args, model)

    if not os.path.exists(args.circuit_dir):
        os.makedirs(args.circuit_dir)

    hist_agg, hist_path = load_hists(args, cfg, save_basename)

    if cfg.seed is not None:
        random.seed(cfg.seed)
    random.shuffle(examples)

    if args.data_type == "hf":
        for i, example in tqdm(enumerate(examples)):
            if not example:
                continue
            example_basename = save_basename + f"_{example[0]['document_idx']}"

            process_examples(
                model,
                embed,
                attns,
                mlps,
                resids,
                dictionaries,
                example_basename,
                example,
                cfg,
                hist_agg,
            )
            if cfg.collect_hists > 0:
                hist_agg.save(hist_path, cfg)
                if i >= cfg.collect_hists:
                    break
    else:
        process_examples(
            model,
            embed,
            attns,
            mlps,
            resids,
            dictionaries,
            save_basename,
            examples,
            cfg,
            hist_agg,
        )


if __name__ == "__main__":
    main()

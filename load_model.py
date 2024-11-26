"""Sparse autoencoder and histogram loading functionality."""

from textwrap import dedent

from nnsight import LanguageModel

from dictionary_learning import AutoEncoder
from dictionary_learning.dictionary import IdentityDict
from histogram_aggregator import NEEDS_HIST, HistAggregator


def load_model_dicts(args, cfg):
    """Load sparse autoencoders."""
    device = cfg.device

    model = LanguageModel(cfg.model, device_map=device, dispatch=True)

    if cfg.model != "gpt2":
        embed = model.gpt_neox.embed_in
        attns = [layer.attention for layer in model.gpt_neox.layers]
        mlps = [layer.mlp for layer in model.gpt_neox.layers]
        resids = [layer for layer in model.gpt_neox.layers]
    else:
        embed = None  # embedding SAE doesn't exist for gpt2
        attns = [layer.attn for layer in model.transformer.h]
        mlps = [layer.mlp for layer in model.transformer.h]
        resids = [layer for layer in model.transformer.h]

    dictionaries = {}

    if cfg.dict_id == "id":
        dictionaries[embed] = IdentityDict(cfg.d_model)
        for i in range(len(model.gpt_neox.layers)):
            dictionaries[resids[i]] = IdentityDict(cfg.d_model)
            dictionaries[mlps[i]] = IdentityDict(cfg.d_model)
            dictionaries[attns[i]] = IdentityDict(cfg.d_model)

    elif cfg.dict_id == "gpt2":
        cfg.update_from_dict(
            dict(
                resid_posn="post",
                parallel_attn=False,
                first_component="attn_0",
                layers=12,
                d_model=768,
            )
        )

        # GPT-2 case.
        for i in range(len(model.transformer.h)):
            # Standardizing resid_post with the rest.
            dictionaries[resids[i]] = AutoEncoder.from_hf(
                "jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs",
                f"v5_128k_layer_{i}.pt/sae_weights.safetensors",
                device=device,
            )
            dictionaries[mlps[i]] = AutoEncoder.from_hf(
                "jbloom/GPT2-Small-OAI-v5-128k-mlp-out-SAEs",
                f"v5_128k_layer_{i}/sae_weights.safetensors",
                device=device,
            )
            dictionaries[attns[i]] = AutoEncoder.from_hf(
                "jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs",
                f"v5_128k_layer_{i}/sae_weights.safetensors",
                device=device,
            )

    else:
        cfg.update_from_dict(
            {
                "resid_posn": "post",
                "parallel_attn": True,
                "first_component": "embed",
                "layers": 6,
                "d_model": 512,
            }
        )
        pre: str = args.dict_path
        post: str = args.dict_id

        # Pythia case.
        dictionaries[embed] = AutoEncoder.from_pretrained(
            f"{pre}/embed/{post}_32768/ae.pt", device=device
        )

        for i in range(len(model.gpt_neox.layers)):
            dictionaries[resids[i]] = AutoEncoder.from_pretrained(
                f"{pre}/resid_out_layer{i}/{post}_32768/ae.pt",
                device=device,
            )
            dictionaries[mlps[i]] = AutoEncoder.from_pretrained(
                f"{pre}/mlp_out_layer{i}/{post}_32768/ae.pt",
                device=device,
            )
            dictionaries[attns[i]] = AutoEncoder.from_pretrained(
                f"{pre}/attn_out_layer{i}/{post}_32768/ae.pt",
                device=device,
            )

    return model, embed, attns, mlps, resids, dictionaries


def load_hists(args, cfg, save_basename):
    """Load neuron-basis activation histograms."""
    hist_agg = HistAggregator(cfg.model)

    needs_hist = (
        cfg.node_thresh_type in NEEDS_HIST
        or cfg.edge_thresh_type in NEEDS_HIST
    )
    default_hist_path = f"histograms/{save_basename}_{cfg.as_fname()}"
    hist_path = (
        args.histogram_path if args.histogram_path else default_hist_path
    )

    if args.accumulate_hists or needs_hist or args.bootstrap_path:
        ret_val = hist_agg.load(
            hist_path
        )  # should return hist_agg if successful

        if ret_val is None and needs_hist:
            raise ValueError(
                dedent(
                    """
                    Threshold method requires histogram, but no existing
                    histogram was found. Please run with --collect_hists first.
                    """
                )
            )

    if cfg.node_thresh_type in NEEDS_HIST:
        hist_agg = hist_agg.cpu()
        hist_agg.compute_node_thresholds(
            cfg.node_threshold, cfg.node_thresh_type
        )

    if cfg.edge_thresh_type in NEEDS_HIST:
        hist_agg = hist_agg.cpu()
        hist_agg.compute_edge_thresholds(
            cfg.edge_threshold, cfg.edge_thresh_type
        )

    if cfg.bootstrap_path:
        hist_agg.reset()
        hist_path = args.bootstrap_path
    return hist_agg, hist_path

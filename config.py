import dataclasses
from typing import Literal
from argparse import Namespace
from enum import Enum

from histogram_aggregator import ThresholdType


@dataclasses.dataclass
class Config:
    # Algorithm controls
    node_threshold: float = 0.0
    edge_threshold: float = 0.0
    node_thresholds: dict[str, float] = (
        None  # mapping from submod name to threshold to be used (typically, we auto-generate this)
    )
    edge_thresholds: dict[dict[str, str], float] = (
        None  # mapping from submod name to threshold to be used (typically, we auto-generate this)
    )
    node_thresh_type: ThresholdType = ThresholdType.THRESH
    edge_thresh_type: ThresholdType = ThresholdType.THRESH
    max_nodes: int = (
        50  # limit feats/submod to at most this many, regardless of thresholding strategy (topk that pass threshold test)
    )
    nodes_only: bool = False
    method: Literal["ig", "attrib", "exact"] = "ig"
    aggregation: Literal["none", "sum"] = "sum"
    prune_method: Literal[
        "none", "source-sink", "sink-backwards", "first-layer-sink"
    ] = "none"
    collect_hists: int = (
        0  # if > 0 then collect histograms for the first collect_hists examples
    )
    bootstrap: bool = (
        False  # if True, an existing histogram will be used to compute thresholds for a given histogram collection run (but we won't write to that one)
    )
    circuit_dir: str = ""  # base path to circuit files
    plot_dir: str = ""  # base path to plots
    bootstrap_path: str = ""  # path to the histogram to use for bootstrapping

    # Data
    dataset: str = "simple_train"
    num_examples: int = 100
    batch_size: int = 32
    example_length: int | None = None
    data_type: Literal["nopair", "regular", "hf"] = "regular"

    # Plotting controls
    edge_thickness_normalization: Literal["linear", "log"] = "linear"
    pen_thickness: float = 0.5
    plot_only: bool = False

    # Model
    model: str = "gpt2"
    d_model: int = 768
    resid_posn: Literal["post", "mid", "pre"] = "post"
    layers: int = 12
    first_component: Literal["embed", "attn_0", "resid_0"] = "attn_0"
    parallel_attn: bool = False
    dict_id: str = "gpt2"
    annotations_path: str = ""

    # Miscellaneous
    device: str = "cuda"
    disable_tqdm: bool = False
    seed: int | None = None

    def update(self, args: Namespace):
        for k, v in vars(args).items():
            if hasattr(self, k):
                setattr(self, k, v)

    def update_from_dict(self, args: dict):
        for k, v in args.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def asdict(self):
        # used for debugging purposes only, don't use this for serialization or loading
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def as_fname(self):
        return f'dict{self.dict_id}_node{self.node_threshold}-{self.node_thresh_type.value}_edge{self.edge_threshold}-{self.edge_thresh_type.value}_agg{self.aggregation}_method{self.method}_prune{self.prune_method}_model{self.model.replace("/", "_")}'

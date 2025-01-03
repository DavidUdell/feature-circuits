from enum import Enum
import json
from typing import TYPE_CHECKING
import dataclasses
import os
import os.path as osp
import glob
from typing import Literal
import warnings
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from safetensors import safe_open
from safetensors.torch import save_file as torch_save_file
from safetensors.numpy import save_file as numpy_save_file

from graph_utils import iterate_edges
if TYPE_CHECKING:
    from config import Config



def normalize_path(path):
    parts = path.split('.')
    try:
        layer = int(parts[-1])   # if the last part is an integer, then we are dealing with a resid
        component = 'resid'
    except ValueError:
        component = parts[-1]
        if component == 'embed_in':
            return 'embed'
        if component == 'attention':
            component = 'attn'
        layer = int(parts[-2])

    return f'{component}_{layer}'

def get_submod_repr(submod):
    """Return canonical string represention of a submod (e.g. attn_0, mlp_5, ...)"""
    if isinstance(submod, str):
        return submod
    if hasattr(submod, '_module_path'):
        path =  submod._module_path
    elif hasattr(submod, 'path'):
        path = submod.path
    else:
        raise ValueError('submod does not have _module_path or path attribute')
    return normalize_path(path)

class ThresholdType(Enum):  # what thresh means in each case
    THRESH = 'thresh'  # raw threshold that activations must exceed
    SPARSITY = 'sparsity'   # the number of features to choose per sequence position
    Z_SCORE = 'z_score'   # how many stds above the mean must the activation be to be considered
    PEAK_MATCH = 'peak_match'  # try to find a raw threshold so that the the thresholded histogram peaks at the thresh, and then decreases after
    PERCENTILE = 'percentile'   # compute a raw threshold based on histograms that will keep the top thresh % of weights

NEEDS_HIST = [ThresholdType.PEAK_MATCH, ThresholdType.PERCENTILE, ThresholdType.Z_SCORE]

class PlotType(Enum):
    REGULAR = 'regular'
    FIRST = 'first'
    ERROR = 'error'
    FIRST_ERROR = 'first_error'

@dataclasses.dataclass
class HistogramSettings:
    n_bins: int = 10_000
    act_min: float = -10
    act_max: float = 5
    nnz_min: float = 0
    n_feats: dict[str, int] = dataclasses.field(default_factory=dict)

class Histogram:
    hist_names = ['nnz', 'acts',
                  'error_nnz', 'error_acts',
                  'first_nnz', 'first_acts',
                  'error_first_nnz', 'error_first_acts']

    def __init__(self, submod: str, settings: HistogramSettings):
        self.submod = submod
        self.settings = settings
        self.thresholds = {}

        self.tracing_nnz = []
        self.tracing_acts = []

        self.reset()

    def state_dict(self):
        tensor_state = {k: getattr(self, k) for k in self.hist_names}
        pyth_state = {'n_samples': self.n_samples, 'submod': self.submod}
        return tensor_state, pyth_state

    def load_state_dict(self, pyth_state, tensor_state):
        for k, v in tensor_state.items():
            setattr(self, k, v)
        self.n_samples = pyth_state['n_samples']
        self.submod = pyth_state['submod']
        return self

    @t.no_grad()
    def compute_hist(self, w: t.Tensor):
        abs_w = abs(w)
        acts_hist = t.histc(t.log10(abs_w[abs_w != 0]), bins=self.settings.n_bins,
                            min=self.settings.act_min, max=self.settings.act_max)
        nnz = (w != 0).sum(dim=2).flatten()
        # this will implicitly ignore cases where there are no non-zero elements
        # since 0s -> -inf through the log10, which is less than the min
        nnz_hist = t.histc(t.log10(nnz), bins=self.settings.n_bins, min=self.settings.nnz_min,
                           max=np.log10(self.settings.n_feats[self.submod]-1)) # -1 to avoid "error" term
        return nnz_hist, acts_hist


    @t.no_grad()
    def compute_hists(self, w: t.Tensor, trace=False):
        self.n_samples += 1

        w_late = w[:, 1:, :-1]
        nnz, acts = self.compute_hist(w_late)

        w_first = w[:, :1, :-1]  # -1 to avoid "error" term
        nnz_first, acts_first = self.compute_hist(w_first)

        w_error = w[:, 1:, -1:]
        nnz_error, acts_error = self.compute_hist(w_error)  # nnz_error is pretty uninformative

        w_first_error = w[:, :1, -1:]
        _, acts_first_error = self.compute_hist(w_first_error)

        if trace:
            self.tracing_nnz.append((nnz.save(), nnz_error.save(), nnz_first.save(), w_first_error.save()))
            self.tracing_acts.append((acts.save(), acts_error.save(), acts_first.save(), acts_first_error.save()))
        else:
            self.nnz += nnz
            self.acts += acts

            self.first_nnz += nnz_first
            self.first_acts += acts_first

            self.error_nnz += nnz_error
            self.error_acts += acts_error

            self.error_first_nnz[0] += (w_first_error == 0).sum()
            self.error_first_nnz[1] += (w_first_error != 0).sum()
            self.error_first_acts += acts_first_error


    def aggregate_traced(self):
        for nnz, nnz_error, nnz_first, w_first_error in self.tracing_nnz:
            self.nnz += nnz.value
            self.error_nnz += nnz_error.value
            self.first_nnz += nnz_first.value
            self.error_first_nnz[0] += (w_first_error.value == 0).sum()
            self.error_first_nnz[1] += (w_first_error.value != 0).sum()

        for acts, acts_error, acts_first, acts_first_error in self.tracing_acts:
            self.acts += acts.value
            self.error_acts += acts_error.value
            self.first_acts += acts_first.value
            self.error_first_acts += acts_first_error.value

        self.tracing_nnz.clear()
        self.tracing_acts.clear()

    def cpu(self):
        tensors = ['nnz', 'acts', 'error_nnz', 'error_acts', 'first_nnz', 'first_acts', 'error_first_nnz', 'error_first_acts']
        for tens in tensors:
            tensor = getattr(self, tens)
            if isinstance(tensor, t.Tensor):
                setattr(self, tens, getattr(self, tens).cpu().numpy())
            setattr(self, tens, getattr(self, tens).astype(np.int64))
        return self

    def reset(self):
        # importantly, we don't reset thresholds
        # the major use case of this reset function is for histogram bootstrapping, where we use one histogram to
        # compute thresholds, and then we want to recompute histograms given these thresholds
        self.n_samples = 0

        self.nnz = t.zeros(self.settings.n_bins, dtype=t.float32).cuda()
        self.acts = t.zeros(self.settings.n_bins, dtype=t.float32).cuda()

        self.error_nnz = t.zeros(self.settings.n_bins, dtype=t.float32).cuda()
        self.error_acts = t.zeros(self.settings.n_bins, dtype=t.float32).cuda()

        self.first_nnz = t.zeros(self.settings.n_bins, dtype=t.float32).cuda()
        self.first_acts = t.zeros(self.settings.n_bins, dtype=t.float32).cuda()

        self.error_first_nnz = t.zeros(2, dtype=t.float32).cuda()  # 2 bins since it can only be 0 or 1
        self.error_first_acts = t.zeros(self.settings.n_bins, dtype=t.float32).cuda()


    def select_hist_type(self, acts_or_nnz, hist_type: PlotType):
        match hist_type:
            case PlotType.REGULAR:
                hist = self.acts if acts_or_nnz == 'acts' else self.nnz
            case PlotType.FIRST:
                hist = self.first_acts if acts_or_nnz == 'acts' else self.first_nnz
            case PlotType.ERROR:
                hist = self.error_acts if acts_or_nnz == 'acts' else self.error_nnz
            case PlotType.FIRST_ERROR:
                hist = self.error_first_acts if acts_or_nnz == 'acts' else self.error_first_nnz
        return hist

    def get_hist_settings(self, acts_or_nnz='acts', hist_type=PlotType.REGULAR, thresh=None, thresh_type=None):

        hist = self.select_hist_type(acts_or_nnz, hist_type)

        if acts_or_nnz == 'acts':
            min_val = self.settings.act_min
            max_val = self.settings.act_max
            xlabel = 'log10(Activation magnitude)'
            bins = np.linspace(min_val, max_val, self.settings.n_bins)
        else:
            min_val = 0
            xlabel = 'NNZ'
            max_val = np.log10(self.settings.n_feats[self.submod])
            bins = 10 ** (np.linspace(min_val, max_val, self.settings.n_bins))
            max_index = np.nonzero(hist)[0].max()
            max_val = bins[max_index]
            bins = bins[:max_index+1]
            hist = hist[:max_index+1]

        if thresh is not None:
            if acts_or_nnz == 'nnz':
                raise ValueError("Cannot compute threshold for nnz")

            thresh_loc = self.get_threshold(bins, acts_or_nnz, hist_type, thresh, thresh_type)
            hist = hist.copy()
            hist[:thresh_loc-1] = 0

        _, median_val, std = self.get_mean_median_std(hist, bins)
        return hist, bins, xlabel, median_val, std

    def get_mean_median_std(self, hist: t.Tensor, bins: t.Tensor):
        total = hist.sum()
        median_idx = (hist.cumsum() >= total / 2).nonzero()[0][0]
        median_val = bins[median_idx]
        mean = (bins * hist).sum() / total
        # compute variance of activations
        std = np.sqrt(((bins - mean)**2 * hist).sum() / total)
        return mean, median_val, std

    def get_threshold(self, bins: t.Tensor, acts_or_nnz: Literal['acts', 'nnz'],
                      hist_type: PlotType, thresh: float, thresh_type: ThresholdType):
        if acts_or_nnz == 'nnz':
            raise ValueError("Cannot compute threshold for nnz")

        hist = self.select_hist_type(acts_or_nnz, hist_type)
        match thresh_type:
            case ThresholdType.THRESH:
                thresh_loc = np.searchsorted(bins, np.log10(thresh))

            case ThresholdType.SPARSITY:
                percentile_hist = np.cumsum(hist) / hist.sum()
                thresh_loc = np.searchsorted(percentile_hist, 1-thresh)

            case ThresholdType.PERCENTILE:
                percentile_hist = np.cumsum(hist) / hist.sum()
                thresh_loc = np.searchsorted(percentile_hist, 1-thresh)

            case ThresholdType.Z_SCORE:
                mean, _, std = self.get_mean_median_std(hist, bins)
                thresh_loc = np.searchsorted(bins, mean + thresh * std)

            case ThresholdType.PEAK_MATCH:
                thresh_loc = None
                best_diff = np.inf
                for b in range(1, len(hist)):
                    thresh_peak = hist[b:].max()
                    hist_diff = np.abs(thresh_peak - thresh)
                    if hist_diff < best_diff:
                        best_diff = hist_diff
                        thresh_loc = b
        return thresh_loc

    def compute_thresholds(self, thresh, thresh_type):
        self.thresholds = {}
        bins = np.linspace(self.settings.act_min, self.settings.act_max, self.settings.n_bins)
        for hist_type in PlotType:
            try:
                self.thresholds[hist_type] = 10**bins[self.get_threshold(bins, 'acts', hist_type, thresh, thresh_type)]
            except IndexError:
                pass
        return self.thresholds

    def threshold(self, w: t.Tensor, ndim: int=3):
        mult = 3 if self.submod == 'resid_11' else 1
        abs_w = abs(w) * mult
        thresh_w = t.zeros_like(abs_w, dtype=t.bool, device='cuda')

        if ndim == 3:
            thresh_w[:, 1:, :-1] = abs_w[:, 1:, :-1] > self.thresholds[PlotType.REGULAR]
            thresh_w[:, :1, :-1] = abs_w[:, :1, :-1] > self.thresholds[PlotType.FIRST]
            thresh_w[:, 1:, -1:] = abs_w[:, 1:, -1:] > self.thresholds[PlotType.ERROR]
            thresh_w[:, :1, -1:] = abs_w[:, :1, -1:] > self.thresholds[PlotType.FIRST_ERROR]
        else:
            thresh_w[:-1] = abs_w[:-1] > self.thresholds[PlotType.REGULAR]
            thresh_w[-1] = abs_w[-1] > self.thresholds[PlotType.ERROR]

        return t.nonzero(thresh_w.flatten()).flatten()


class HistAggregator:
    def __init__(self, model_str='gpt2', n_bins=10_000, act_min=-10, act_max=5):
        self.settings = HistogramSettings(n_bins=n_bins, act_min=act_min, act_max=act_max)
        self.model_str = model_str

        self.nodes: dict[str, Histogram] = {}
        self.edges: dict[dict[str, Histogram]] = {}


    @t.no_grad()
    def compute_node_hist(self, submod, w: t.Tensor):
        # w: [N, seq_len, n_feats]
        submod = get_submod_repr(submod)
        if submod == 'resid_11':
            pass
        if submod not in self.nodes:
            self.settings.n_feats[submod] = w.shape[2]
            self.nodes[submod] = Histogram(submod, self.settings)

        self.nodes[submod].compute_hists(w)


    def get_edge_hist(self, up_submod, down_submod) -> Histogram:
        up_submod = get_submod_repr(up_submod)
        down_submod = get_submod_repr(down_submod)

        if up_submod not in self.edges:
            self.edges[up_submod] = {}

        if down_submod not in self.edges[up_submod]:
            self.edges[up_submod][down_submod] = Histogram(up_submod, self.settings)

        return self.edges[up_submod][down_submod]

    def cpu(self):
        for n in self.nodes.values():
            n.cpu()
        for up in self.edges:  # pylint: disable=consider-using-dict-items
            for e in self.edges[up].values():
                e.cpu()
        return self

    def reset(self):
        """Reset histograms to zero. Importantly, thresholds are not reset,
        so that boostrapping works."""
        for n in self.nodes.values():
            n.reset()
        for up in self.edges:  # pylint: disable=consider-using-dict-items
            for e in self.edges[up].values():
                e.reset()
        return self

    def save(self, path: str, cfg: 'Config'):
        """
        Save histogram to disk. Path corresponds to a directory, where we will
        store 2 files, the tensor state in a safetensor and the state of config
        objects in a json file.
        """
        pyth_state = {
            'settings': dataclasses.asdict(self.settings),
            'model_str': self.model_str,
            'cfg': cfg.asdict()  # save config purely for debugging purposes
        }

        # save tensor state with safetensors, and save pyth state as a json to path.cfg
        os.makedirs(path, exist_ok=True)

        if isinstance(next(iter(self.nodes.values())).nnz, t.Tensor):
            save_file = torch_save_file
        else:
            save_file = numpy_save_file

        for node, hist in self.nodes.items():
            pyth_state[node] = {}
            tensor_state, pyth_state[node]['node'] = hist.state_dict()
            os.makedirs(osp.join(path, node), exist_ok=True)
            save_file(tensor_state, osp.join(path, node, 'node.safetensors'))

        for up in self.edges:  # pylint: disable=consider-using-dict-items
            for down, hist in self.edges[up].items():
                tensor_state, pyth_state[up][down] = hist.state_dict()
                save_file(tensor_state, osp.join(path, up, f'{down}.safetensors'))

        with open(osp.join(path, 'config.json'), 'w') as f:
            json.dump(pyth_state, f, indent=4)

    def load_into_hist(self, path, map_location, submod, pyth_state):
        tensor_state = {}
        with safe_open(path, 'pt', device=map_location) as f:
            for k in f.keys():
                tensor_state[k] = f.get_tensor(k)
        return Histogram(submod, self.settings).load_state_dict(pyth_state, tensor_state)

    def load(self, path, map_location=None):
        if not osp.exists(path):
            warnings.warn(f'Tried to load histograms, but path "{path}" was not found...')
            return None

        with open(osp.join(path, 'config.json'), 'r') as f:
            pyth_state = json.load(f)

        self.settings = HistogramSettings(**pyth_state['settings'])
        self.model_str = pyth_state['model_str']

        for node in glob.glob(osp.join(path, '*')):
            if osp.isdir(node):
                submod = osp.basename(node)
                # load node histograms
                self.nodes[submod] = self.load_into_hist(osp.join(node, 'node.safetensors'),
                                                         map_location, submod, pyth_state[submod]['node'])
                self.edges[submod] = {}

                # load edge histograms
                for edge in glob.glob(osp.join(node, '*.safetensors')):
                    down = osp.splitext(osp.basename(edge))[0]
                    self.edges[submod][down] = self.load_into_hist(edge, map_location, submod, pyth_state[submod][down])

        print("Successfully loaded histograms at", path)
        return self

    def get_hist_for_node_effect(self, layer, component, acts_or_nnz, plot_type: PlotType, thresh=None, thresh_type=None):
        mod_name = f'{component}_{layer}'
        hist = self.nodes[mod_name]
        return hist.get_hist_settings(acts_or_nnz, plot_type, thresh, thresh_type)

    def get_hist_for_edge_effect(self, up:str, down: str, acts_or_nnz, plot_type: PlotType, thresh=None, thresh_type=None):
        hist = self.edges[up][down]#.select_hist_type(acts_or_nnz, plot_type)
        return hist.get_hist_settings(acts_or_nnz, plot_type, thresh, thresh_type)

    def plot_hist(self, hist, median, std, bins, ax, xlabel, title):
        value_hist_color = 'blue'
        ax.set_xlabel(xlabel, color=value_hist_color)
        ax.set_ylabel('Frequency', color=value_hist_color)
        ax.plot(bins, hist, color=value_hist_color)
        ax.tick_params(axis='x', colors=value_hist_color)
        ax.tick_params(axis='y', colors=value_hist_color)
        # ax.set_xlim(min(min_nnz, min_val), max(max_nnz, max_val))
        # compute median value of activations
        ax.set_title(f'{title}')
        # vertical line at mean
        ax.axvline(median, color='r', linestyle='--')
        # add text with mean
        ax.text(median+0.5, hist.max(), f'{median:.2f} +- {std:.2f}', color='r')

    def compute_edge_thresholds(self, thresh: float, thresh_type: ThresholdType):
        for up in self.edges:
            for hist in self.edges[up].values():
                hist.compute_thresholds(thresh, thresh_type)

    def compute_node_thresholds(self, thresh: float, thresh_type: ThresholdType):
        for mod_name in self.nodes:
            hist = self.nodes[mod_name]
            hist.compute_thresholds(thresh, thresh_type)

    def plot(self, n_layers, nodes_or_edges: Literal['nodes', 'edges']='nodes',
             acts_or_nnz:Literal['acts', 'nnz'] ='acts', plot_type: PlotType = PlotType.REGULAR,
             thresh=None, thresh_type=ThresholdType.THRESH):

        if nodes_or_edges == 'nodes':
            fig, axs = plt.subplots(n_layers, 3, figsize=(18, 3.6*n_layers))

            for layer in range(n_layers):
                for i, component in enumerate(['resid', 'attn', 'mlp']):
                    hist, bins, xlabel, median, std = self.get_hist_for_node_effect(layer, component, acts_or_nnz,
                                                                                    plot_type, thresh, thresh_type)
                    self.plot_hist(hist, median, std, bins, axs[layer, i], xlabel, f'{self.model_str} {component} layer {layer}')

        elif nodes_or_edges == 'edges':
            edges_per_layer = 6 if self.model_str == 'gpt2' else 5
            first_component = 'attn_0' if self.model_str == 'gpt2' else 'embed'
            fig, axs = plt.subplots(n_layers, edges_per_layer, figsize=(6*edges_per_layer, 3.6*n_layers))


            for layer in range(n_layers):
                for i, (up, down) in enumerate(iterate_edges(self.edges, layer, first_component)):
                    hist, bins, xlabel, median, std = self.get_hist_for_edge_effect(up, down, acts_or_nnz, plot_type, thresh, thresh_type)
                    self.plot_hist(hist, median, std, bins, axs[layer, i], xlabel, f'{self.model_str} layer {layer} edge {(up, down)}')


        plt.tight_layout()
        plt.show()

import os
import sys
from torch.utils import data
import pickle

class AllGraphDataSampler(data.Dataset):
    def __init__(
        self,
        base_dir,
        gname_list=None,
        data_start=None,
        data_middle=None,
        data_end=None,
        mode="train",
        purge_gap=None,
    ):
        self.data_dir = os.path.join(base_dir)
        self.mode = mode
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        if gname_list is None:
            self.gnames_all = os.listdir(self.data_dir)
            self.gnames_all.sort()
        if purge_gap is None:
            self.purge_gap = self._infer_purge_gap()
        else:
            self.purge_gap = max(0, int(purge_gap))
        if mode == "train":
            train_end = max(self.data_start, self.data_middle - self.purge_gap)
            self.gnames_all = self.gnames_all[self.data_start:train_end]
        elif mode == "val":
            self.gnames_all = self.gnames_all[self.data_middle:self.data_end]
        self.data_all = self.load_state()

    def __len__(self):
        return len(self.data_all)

    def _infer_purge_gap(self):
        if self.data_middle is None or self.data_middle <= 0:
            return 0
        if self.data_middle > len(self.gnames_all):
            return 0
        sample_name = self.gnames_all[self.data_middle - 1]
        sample_path = os.path.join(self.data_dir, sample_name)
        try:
            sample = pickle.load(open(sample_path, "rb"))
            labels = sample.get("labels")
            if labels is None:
                return 0
            if hasattr(labels, "shape") and len(labels.shape) > 1:
                horizon = int(labels.shape[-1])
            else:
                horizon = 1
            return max(0, horizon - 1)
        except Exception:
            return 0

    def load_state(self):
        data_all = []
        length = len(self.gnames_all)
        skipped = 0
        for i in range(length):
            sys.stdout.flush()
            sys.stdout.write('{} data loading: {:.2f}%{}'.format(self.mode, i*100/length, '\r'))
            sample_path = os.path.join(self.data_dir, self.gnames_all[i])
            try:
                sample = pickle.load(open(sample_path, "rb"))
            except Exception:
                skipped += 1
                continue
            if not self._is_valid_sample(sample):
                skipped += 1
                continue
            data_all.append(sample)
        print('{} data loaded!'.format(self.mode))
        if skipped > 0:
            print(f'{self.mode} skipped invalid samples: {skipped}')
        return data_all

    @staticmethod
    def _is_valid_sample(sample):
        if not isinstance(sample, dict):
            return False
        required_keys = ["features", "labels", "pos_adj", "neg_adj", "mask"]
        for key in required_keys:
            if key not in sample:
                return False
        features = sample["features"]
        labels = sample["labels"]
        pos_adj = sample["pos_adj"]
        neg_adj = sample["neg_adj"]
        for tensor_like in [features, labels, pos_adj, neg_adj]:
            if not hasattr(tensor_like, "shape"):
                return False
        if hasattr(features, "numel") and features.numel() == 0:
            return False
        if hasattr(labels, "numel") and labels.numel() == 0:
            return False
        if hasattr(pos_adj, "numel") and pos_adj.numel() == 0:
            return False
        if hasattr(neg_adj, "numel") and neg_adj.numel() == 0:
            return False
        return True

    def __getitem__(self, idx):
        return self.data_all[idx]

import numpy as np
from typing import Optional, List, Dict


class AverageMeter:
    def __init__(self, last_n=None):
        self._records = []
        self.last_n = last_n

    def update(self, result):
        if isinstance(result, (list, tuple)):
            self._records += result
        else:
            self._records.append(result)

    def reset(self):
        self._records.clear()

    @property
    def records(self):
        if self.last_n is not None:
            return self._records[-self.last_n :]
        else:
            return self._records

    def sum(self):
        return np.sum(self.records)

    def mean(self):
        return np.mean(self.records)

    def std(self):
        return np.std(self.records)

    def median(self):
        return np.median(self.records)


class SummaryBoard:
    def __init__(self, names: Optional[List[str]] = None, last_n: Optional[int] = None, adaptive=False):
        r"""Instantiate a SummaryBoard.

        Args:
            names (List[str]=None): create AverageMeter with the names.
            last_n (int=None): only the last n records are used.
            adaptive (bool=False): whether register basic meters automatically on the fly.
        """
        self.meter_names: List[str] = []
        self.meter_dict: Dict[str, AverageMeter] = {}
        self.last_n = last_n
        self.adaptive = adaptive

        if names is not None:
            self.register_all(names)

    def register_meter(self, name):
        self.meter_dict[name] = AverageMeter(last_n=self.last_n)
        self.meter_names.append(name)

    def register_all(self, names):
        for name in names:
            self.register_meter(name)

    def reset_meter(self, name):
        self.meter_dict[name].reset()

    def reset_all(self):
        for name in self.meter_names:
            self.reset_meter(name)

    def check_name(self, name):
        if name not in self.meter_names:
            if self.adaptive:
                self.register_meter(name)
            else:
                raise KeyError('No meter for key "{}".'.format(name))

    def update(self, name, value):
        self.check_name(name)
        self.meter_dict[name].update(value)

    def update_from_dict(self, result_dict):
        if not isinstance(result_dict, dict):
            raise TypeError('`result_dict` must be a dict: {}.'.format(type(result_dict)))
        for key, value in result_dict.items():
            if key not in self.meter_names and self.adaptive:
                self.register_meter(key)
            if key in self.meter_names:
                self.meter_dict[key].update(value)

    def sum(self, name):
        self.check_name(name)
        return self.meter_dict[name].sum()

    def mean(self, name):
        self.check_name(name)
        return self.meter_dict[name].mean()

    def std(self, name):
        self.check_name(name)
        return self.meter_dict[name].std()

    def median(self, name):
        self.check_name(name)
        return self.meter_dict[name].median()

    def summary(self, names=None):
        if names is None:
            names = self.meter_names
        summary_dict = {name: self.meter_dict[name].mean() for name in names}
        return summary_dict
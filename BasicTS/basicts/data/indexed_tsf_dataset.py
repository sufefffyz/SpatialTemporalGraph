from .coreset_tsf_dataset import CoresetTimeSeriesForecastingDataset
from .simple_tsf_dataset import TimeSeriesForecastingDataset


class IndexedTimeSeriesForecastingDataset(TimeSeriesForecastingDataset):
    """Forecasting dataset that appends the sample start index."""

    def resolve_index(self, index: int) -> int:
        return int(index)

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        sample["index"] = self.resolve_index(index)
        return sample


class IndexedCoresetTimeSeriesForecastingDataset(CoresetTimeSeriesForecastingDataset):
    """Coreset dataset that appends the original train-window start index."""

    def resolve_index(self, index: int) -> int:
        if hasattr(self, "selected_indices"):
            return int(self.selected_indices[index])
        return int(index)

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        sample["index"] = self.resolve_index(index)
        return sample

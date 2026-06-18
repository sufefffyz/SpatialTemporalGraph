import sys
from pathlib import Path
import unittest

import torch


BASICTS_ROOT = Path(__file__).resolve().parents[3]
if str(BASICTS_ROOT) not in sys.path:
    sys.path.insert(0, str(BASICTS_ROOT))

from baselines.STID.arch import STID  # noqa: E402


class TestSTIDSubgraphForward(unittest.TestCase):
    def build_model(self):
        return STID(
            num_nodes=6,
            input_len=2,
            input_dim=3,
            embed_dim=4,
            output_len=1,
            num_layer=1,
            if_node=True,
            node_dim=3,
            if_T_i_D=False,
            if_D_i_W=False,
            temp_dim_tid=0,
            temp_dim_diw=0,
            time_of_day_size=4,
            day_of_week_size=7,
        )

    def test_idx_none_preserves_full_node_output_shape(self):
        model = self.build_model()
        history = torch.randn(2, 2, 6, 3)
        future = torch.randn(2, 1, 6, 3)

        output = model(
            history_data=history,
            future_data=future,
            batch_seen=0,
            epoch=1,
            train=True,
            idx=None,
        )

        self.assertEqual(tuple(output.shape), (2, 1, 6, 1))

    def test_idx_selects_matching_node_embeddings_for_subgraph_forward(self):
        model = self.build_model()
        node_idx = torch.tensor([1, 3, 5], dtype=torch.long)
        history = torch.randn(2, 2, 6, 3)
        future = torch.randn(2, 1, 6, 3)
        sub_history = history[:, :, node_idx, :]
        sub_future = future[:, :, node_idx, :]

        output = model(
            history_data=sub_history,
            future_data=sub_future,
            batch_seen=0,
            epoch=1,
            train=True,
            idx=node_idx,
        )

        self.assertEqual(tuple(output.shape), (2, 1, 3, 1))


if __name__ == "__main__":
    unittest.main()

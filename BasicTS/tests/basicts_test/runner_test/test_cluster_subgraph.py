import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


UTILS_PATH = (
    Path(__file__).resolve().parents[3]
    / "basicts"
    / "runners"
    / "runner_zoo"
    / "cluster_subgraph_utils.py"
)
UTILS_SPEC = importlib.util.spec_from_file_location("cluster_subgraph_utils", UTILS_PATH)
UTILS = importlib.util.module_from_spec(UTILS_SPEC)
assert UTILS_SPEC.loader is not None
sys.modules[UTILS_SPEC.name] = UTILS
UTILS_SPEC.loader.exec_module(UTILS)

build_balanced_spatial_kdtree_clusters = UTILS.build_balanced_spatial_kdtree_clusters
build_balanced_signal_kmeans_clusters = UTILS.build_balanced_signal_kmeans_clusters
build_random_balanced_clusters = UTILS.build_random_balanced_clusters
ClusterSubgraphScheduler = UTILS.ClusterSubgraphScheduler
reduce_node_scores_to_samples = UTILS.reduce_node_scores_to_samples


class TestClusterSubgraphUtils(unittest.TestCase):
    def assertBalancedPartition(self, clusters, num_nodes, num_clusters):
        flattened = sorted(int(node) for cluster in clusters for node in cluster)
        sizes = [len(cluster) for cluster in clusters]

        self.assertEqual(flattened, list(range(num_nodes)))
        self.assertEqual(len(clusters), num_clusters)
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_spatial_kdtree_clusters_cover_nodes_once_and_are_balanced(self):
        coords = np.asarray(
            [
                [x, y]
                for x in range(4)
                for y in range(4)
            ],
            dtype=np.float32,
        )

        assignment = build_balanced_spatial_kdtree_clusters(
            coords,
            num_clusters=8,
            dataset_name="unit",
            seed=7,
        )

        self.assertEqual(assignment.cluster_type, "spatial_kdtree")
        self.assertEqual(assignment.num_clusters, 8)
        self.assertEqual(assignment.dataset_name, "unit")
        self.assertEqual(assignment.seed, 7)
        self.assertBalancedPartition(assignment.clusters, num_nodes=16, num_clusters=8)

    def test_signal_kmeans_clusters_cover_nodes_once_and_are_balanced(self):
        steps_per_day = 4
        days = 6
        num_nodes = 16
        time = np.arange(steps_per_day, dtype=np.float32)
        patterns = np.stack(
            [
                np.sin(time),
                np.cos(time),
                time / steps_per_day,
                (steps_per_day - time) / steps_per_day,
            ],
            axis=0,
        )
        data = np.zeros((steps_per_day * days, num_nodes, 1), dtype=np.float32)
        for node in range(num_nodes):
            pattern = patterns[node % len(patterns)]
            data[:, node, 0] = np.tile(pattern, days) + node * 1e-4

        assignment = build_balanced_signal_kmeans_clusters(
            data,
            num_clusters=8,
            dataset_name="unit",
            seed=11,
            steps_per_day=steps_per_day,
            max_iter=20,
        )

        self.assertEqual(assignment.cluster_type, "signal_kmeans")
        self.assertBalancedPartition(assignment.clusters, num_nodes=num_nodes, num_clusters=8)

    def test_random_balanced_clusters_are_seeded_and_balanced(self):
        first = build_random_balanced_clusters(num_nodes=17, num_clusters=8, dataset_name="unit", seed=3)
        second = build_random_balanced_clusters(num_nodes=17, num_clusters=8, dataset_name="unit", seed=3)

        self.assertEqual(first.clusters, second.clusters)
        self.assertEqual(first.cluster_type, "random_balanced_clusters")
        self.assertBalancedPartition(first.clusters, num_nodes=17, num_clusters=8)

    def test_scheduler_balances_cluster_visits_within_epoch(self):
        assignment = build_random_balanced_clusters(num_nodes=16, num_clusters=8, dataset_name="unit", seed=5)
        scheduler = ClusterSubgraphScheduler(assignment=assignment, num_active_clusters=2, seed=13)

        scheduler.begin_epoch(epoch=1, force_full=False)
        visits = {cluster_id: 0 for cluster_id in range(8)}
        for step in range(4):
            selection = scheduler.select(step)
            self.assertEqual(len(selection.cluster_ids), 2)
            for cluster_id in selection.cluster_ids:
                visits[cluster_id] += 1

        self.assertEqual(set(visits.values()), {1})
        self.assertAlmostEqual(selection.active_node_ratio, 0.25)

    def test_scheduler_force_full_returns_all_nodes_for_final_annealing(self):
        assignment = build_random_balanced_clusters(num_nodes=16, num_clusters=8, dataset_name="unit", seed=5)
        scheduler = ClusterSubgraphScheduler(assignment=assignment, num_active_clusters=2, seed=13)

        scheduler.begin_epoch(epoch=99, force_full=True)
        selection = scheduler.select(iter_index=0)

        self.assertEqual(selection.node_indices, list(range(16)))
        self.assertEqual(selection.cluster_ids, list(range(8)))
        self.assertAlmostEqual(selection.active_node_ratio, 1.0)

    def test_reduce_node_scores_to_samples_uses_active_nodes_only(self):
        node_scores = torch.tensor(
            [
                [1.0, 10.0, 100.0, 1000.0],
                [2.0, 20.0, 200.0, 2000.0],
            ]
        )

        reduced = reduce_node_scores_to_samples(node_scores, torch.tensor([1, 3]))

        self.assertTrue(torch.allclose(reduced, torch.tensor([505.0, 1010.0])))


if __name__ == "__main__":
    unittest.main()

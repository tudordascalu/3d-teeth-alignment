import trimesh
import numpy as np
import os
import unittest

from scripts.utils.mesh_saver import MeshSaver


class TestMeshSaver(unittest.TestCase):
    def setUp(self):
        # Define test data
        self.mesh = trimesh.creation.box()
        self.tooth_labels = np.array([1, 2, 0, 1, 2, 0, 3, 0, 3])
        self.id = 123
        self.jaw = 'upper'
        self.colors = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.saver = MeshSaver(self.colors)
        self.filename = f"../../output/{self.id}_gt_{self.jaw}.obj"

    def test_save_processed_mesh(self):
        # Call save_processed_mesh method
        self.saver(self.mesh, self.tooth_labels, self.filename)
        # Check that output file exists
        self.assertTrue(os.path.exists(self.filename))
        # Check that output file is not empty
        self.assertGreater(os.path.getsize(self.filename), 0)
        # Check that output file can be loaded as a mesh
        loaded_mesh = trimesh.load(self.filename)
        self.assertIsInstance(loaded_mesh, trimesh.Trimesh)

    def tearDown(self):
        # Remove test output files
        if os.path.exists(self.filename):
            os.remove(self.filename)

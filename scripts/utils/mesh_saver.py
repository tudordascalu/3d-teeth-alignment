class MeshSaver:
    def __init__(self, colors):
        self.colors = colors

    def __call__(self, mesh, tooth_labels, file_path):
        mesh_processed = mesh.copy()
        # Color mesh
        mesh_processed.visual.vertex_colors = [self.colors[label] for label in tooth_labels]
        # Remove gums and missing teeth
        mask = tooth_labels != 0
        face_mask = mask[mesh_processed.faces].all(axis=1)
        mesh_processed.update_faces(face_mask)
        # Save mesh
        mesh_processed.export(file_path)

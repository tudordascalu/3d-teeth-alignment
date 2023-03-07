class MeshSaver:
    def __init__(self, colors):
        self.colors = colors

    def __call__(self, mesh, tooth_labels, file_path):
        # Color mesh
        mesh.visual.vertex_colors = [self.colors[label] for label in tooth_labels]
        # Remove gums and missing teeth
        mask = tooth_labels != 0
        face_mask = mask[mesh.faces].all(axis=1)
        mesh.update_faces(face_mask)
        # Save mesh
        mesh.export(file_path)

import numpy as np
import torch
import pytorch_lightning as pl
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import softmax
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau


class AlignmentNet(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-8, scheduler_patience=10):
        super(AlignmentNet, self).__init__()
        self.save_hyperparameters()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, (1, 3), padding="same", bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, (1, 3), padding="same", bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 2)),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 17, (1, 3), padding="same", bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((1, 4)),
            torch.nn.BatchNorm2d(17),
        )

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(17 * 17, 17 * 17),
            torch.nn.ReLU()
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.reshape(-1, 17, 17)
        x = (softmax(x, dim=-1) + softmax(x, dim=-2)) / 2
        return x

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(),
                            lr=self.lr,
                            weight_decay=self.weight_decay,
                            momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=self.scheduler_patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        accuracy = self._accuracy(y_pred, y)
        self.log("train_loss", loss)
        return {"loss": loss, "train_loss": loss, "train_accuracy": accuracy}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["train_accuracy"] for x in outputs]).mean()
        self.log("step", self.trainer.current_epoch)
        self.log("avg_loss", {"train": avg_loss})
        self.log("avg_accuracy", {"train": avg_accuracy})

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        accuracy = self._accuracy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("step", self.trainer.current_epoch)
        self.log("avg_loss", {"val": avg_loss})
        self.log("avg_accuracy", {"val": avg_accuracy})

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        accuracy = self._accuracy(y_pred, y)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["test_accuracy"] for x in outputs]).mean()
        self.log("step", self.trainer.current_epoch)
        self.log("avg_loss", {"test": avg_loss})
        self.log("avg_accuracy", {"test": avg_accuracy})
        self.log("test_loss", avg_loss)

    @staticmethod
    def _loss(y_pred, y_true):
        """
        The loss is applied both column-wise and row-wise in order to promote single-class predictions.

        :param y_pred: torch.Tensor of shape (n_batch_size, 17, 17)
        :param y_true: torch.Tensor of shape (n_batch_size, 17, 17)
        :return: torch.Tensor featuring loss value
        """
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(y_pred.reshape(-1, 17), y_true.reshape(-1, 17))
        # Uncomment if we softmax activation is not applied previously
        # y_pred_row = softmax(y_pred, dim=-1).reshape(-1, 17)
        # y_pred_col = softmax(y_pred, dim=-2).reshape(-1, 17)
        # y_true_processed = y_true.reshape(-1, 17)
        # loss = mse_loss(y_pred_row, y_true_processed) + mse_loss(y_pred_col, y_true_processed)
        return loss

    @staticmethod
    def _accuracy(y_pred, y_true):
        """
        Accuracy is computed as the total number of correctly identified teeth.
        We can also consider computing it as total number of correctly identified misaligned teeth, as we have fewer cases.

        :param y_pred: torch.Tensor of shape (n_batch_size, 17, 17)
        :param y_true: torch.Tensor of shape (n_batch_size, 17, 17)
        :return: total number of correctly identified teeth / total number of teeth
        """
        y_pred_processed, y_true_processed = y_pred.argmax(-1).reshape(-1), y_true.argmax(-1).reshape(-1)
        return (y_pred_processed == y_true_processed).sum() / len(y_pred_processed)

# class CentroidMapper:
#     """
#     Maps centroids given a set of labels.
#     """
#
#     def __init__(self, n_teeth=17):
#         """
#         :param n_teeth: number of teeth
#         """
#         self.n_teeth = n_teeth
#
#     def __call__(self, mesh, instances):
#         """
#         Computes the centroid of each tooth instance.
#
#         :param mesh: trimesh.Trimesh :param instances: np.array of shape (len(mesh.vertices),) :return: np.array of
#         shape (n_teeth, 3) featuring the centroid of each tooth; If tooth is missing, centroid coordinates are (0, 0,
#         0).
#         """
#         # Compute vertices
#         vertices = mesh.vertices
#         # Compute unique tooth labels
#         instances_unique = np.unique(instances)
#         instances_unique = np.delete(instances_unique, instances_unique == 0)
#         centroids = np.zeros((self.n_teeth, 3))
#         for instance in instances_unique:
#             centroids[instance - 1] = np.mean(vertices[np.where(instance == instances)], axis=0)
#         return centroids

#
# class Aligner:
#     """
#     This class uses a trained AlignmentNet instance to align a set of tooth labels.
#     """
#
#     def __init__(self, model, distance_means, distance_stds, n_teeth=17):
#         """
#         :param model: instance of AlignmentNet
#         :param distance_means: np.array of shape (17, 17) featuring mean distances between tooth-tooth pairs
#         :param distance_stds: np.array of shape (17, 17) featuring stds for tooth-tooth pair distances
#         :param n_teeth: controls how many double teeth should be accounted for
#         """
#         self.model = model
#         self.distance_means = distance_means
#         self.distance_stds = distance_stds
#         self.centroid_mapper = CentroidMapper(n_teeth)
#         self.distance_mapper = DistanceMapper(n_teeth)
#         self.p_distance_mapper = PDistanceMapper(n_teeth)
#
#     def __call__(self, mesh, instances, double_tooth_label=0):
#         """
#         This method computes teeth features (coming soon) and a probability distance map from the labels. Next,
#         it passes the labels through the model. If there are teeth that have been re-aligned, we compute combinations
#         of all possible re-alignments and pass them through the network with the aim of finding the combination that
#         yields the larger determinants (lower uncertainty).
#
#         :param mesh: trimesh.Trimesh
#         :param instances: np.array of shape (n,), where n corresponds to the total number of vertices in the mesh
#         :param double_tooth_label: label of double tooth to be used for distance map
#         :return: np.array featuring labels post alignment
#         """
#         # Compute centroids
#         centroids = self.centroid_mapper(mesh, instances)
#         # Compute unique instances
#         instances_unique = np.unique(instances)
#         instances_unique = np.delete(instances_unique, instances_unique == 0)
#         # Align instances
#         instances_unique_aligned = self._align(centroids, instances_unique, double_tooth_label)
#         # Re-arrange instances
#         instances_aligned = np.zeros(instances.shape, dtype=np.int32)
#         for instance_old, instance_new in zip(np.arange(17), instances_unique_aligned):
#             instances_aligned[instances == instance_old] = instance_new
#         return instances_aligned
#
#     def _align(self, centroids, instances_unique, double_tooth_label=0):
#         """
#         This method computes teeth features (coming soon) and a probability distance map from centroids and instances.
#
#         :param centroids: np.array of shape (n_teeth, 3) featuring tooth centroids
#         :param instances_unique: np.array of shape (n_detected_teeth,)
#         :param double_tooth_label: label of double tooth to be used for distance map
#         :return: np.array of shape (n_teeth, ) featuring tooth instances aligned
#         """
#         # Compute distance map
#         distance_map = self.distance_mapper(instances_unique, centroids)
#         # Compute probability map based on distances
#         distance_means = np.concatenate(
#             [self.distance_means, np.expand_dims(self.distance_means[:, double_tooth_label], axis=1)], axis=1)
#         distance_stds = np.concatenate(
#             [self.distance_stds, np.expand_dims(self.distance_stds[:, double_tooth_label], axis=1)], axis=1)
#         p_distance_map = self.p_distance_mapper(instances_unique, distance_map, distance_means, distance_stds)
#         # Convert to tensor
#         p_distance_map = torch.from_numpy(p_distance_map).permute(2, 0, 1).type(torch.float32)
#         # Compute network prediction
#         output = self.model(p_distance_map.unsqueeze(0)).squeeze().detach().cpu().numpy()
#         # Solve for unique solution using scipy's implementation of the assignment problem
#         _, instances_aligned = linear_sum_assignment(output, maximize=True)
#         return instances_aligned

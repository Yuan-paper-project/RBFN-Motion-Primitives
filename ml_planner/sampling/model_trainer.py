__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import os
from typing import Callable
import json
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from ml_planner.sampling.networks.layers.basis_functions import basis_func_dict


# Define the Model Trainer Class
class ModelTrainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model_path: str,
        model: Callable,
        model_kwargs: dict,
        loss_functions: list[Callable],
        device: str = "cuda",
        log_dir: str = "",
        optimizer: Callable = torch.optim.Adam,
        opti_kwargs: dict = {"lr": 1e-3},
        draw_network_graph: bool = False,
        draw_trajectories_every_i_epoch: int = 10,
    ):
        # validate input
        self._validate_input(model_kwargs, train_dataloader)
        self.model_path = model_path
        self.draw_trajectories = draw_trajectories_every_i_epoch

        # datasets
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.dataset_input_idxs = [
            train_dataloader.dataset.dataset.input_labels.index(i) for i in model_kwargs["input_labels"]
        ]
        self.dataset_output_idxs = [
            train_dataloader.dataset.dataset.labels.index(i) for i in model_kwargs["output_labels"]
        ]
        print(f"Training on {len(self.train_dataloader.dataset)} samples")
        print(f"Testing on {len(self.test_dataloader.dataset)} samples")
        print(f"batch size: {self.train_dataloader.batch_size}")

        # Get cpu, gpu or mps device for training.
        self.device = (
            "cuda"
            if (device == "cuda" and torch.cuda.is_available())
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")
        # Create Model
        model_kwargs["bounds"] = model_kwargs["bounds"][self.dataset_output_idxs].tolist()
        self.model = model(**model_kwargs).to(self.device)
        print(self.model)

        # Create and save config of model
        path = f"{self.model_path}/config.json"

        model_kwargs.update(self.train_dataloader.dataset.dataset.dataparams)
        with open(path, "w") as f:
            json.dump(model_kwargs, f)

        # Define loss function and optimizer for the training
        self.reduction = "mean"
        self.loss_fns = [
            loss_fn(labels=self.model.output_labels, reduction=self.reduction) for loss_fn in loss_functions
        ]
        self.optimizer = optimizer(self.model.parameters(), **opti_kwargs)

        # tensorboard logging
        log_dir = self.model_path / log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # add graphs
        if self.model.num_rbf_heads < 1000 and draw_network_graph:
            example_input, _ = next(iter(self.train_dataloader))
            self.writer.add_graph(self.model, example_input.to(self.device))

    def _validate_input(self, model_kwargs, train_dataloader):
        basis_funcs = basis_func_dict().values()
        assert (
            model_kwargs["basis_func"] in basis_func_dict
        ), f"Unknown basis function <{model_kwargs['basis_func']}>, only <{basis_funcs}> are supported"
        labels = train_dataloader.dataset.dataset.input_labels
        assert all(
            [i in labels for i in model_kwargs["input_labels"]]
        ), f"Unknown input labels <{model_kwargs['input_labels']}>, only <{labels}> are supported"
        labels = train_dataloader.dataset.dataset.labels
        assert all(
            [i in labels for i in model_kwargs["output_labels"]]
        ), f"Unknown output labels <{model_kwargs['output_labels']}>, only <{labels}> are supported"

    def train(self, epoch: int):
        """Training routine for the model"""
        self.model.train()
        running_losses = [0.0] * (len(self.loss_fns) + 1)
        total_steps = len(self.train_dataloader) - 1

        pbar = tqdm.tqdm(
            enumerate(self.train_dataloader),
            total=total_steps,
            position=1,
            leave=False,
            desc="Sample",
            ncols=100,
            disable=False,
        )

        for idx, (input_data, gt_data) in pbar:
            # get data of used labels
            input_vec = input_data[..., self.dataset_input_idxs].to(self.device)
            gt_vec = gt_data[..., self.dataset_output_idxs].to(self.device)

            # Compute prediction
            pred = self.model(input_vec)

            # Calculate losses
            losses = [loss_fn(pred, gt_vec) for loss_fn in self.loss_fns]
            total_loss = sum(losses)
            running_losses[0] += total_loss.detach().item()
            for i, loss in enumerate(losses):
                running_losses[i + 1] += loss.detach().item()

            # Backpropagation
            total_loss.backward()
            if idx < 4:
                for tag, value in self.model.named_parameters():
                    if value.grad is not None:
                        self.writer.add_histogram(f"Grad/batch{idx}/{tag}", value.grad, epoch + 1)
                        self.writer.add_scalar(f"Grad_norm/batch{idx}/{tag}", value.grad.norm().item(), epoch + 1)
                        self.writer.add_scalar(f"Grad_mean/batch{idx}/{tag}", value.grad.mean().item(), epoch + 1)

                self.writer.flush()
            self.optimizer.step()

            self.optimizer.zero_grad()

            # Visualize current predictions
            if ((epoch + 1) % self.draw_trajectories == 0 or epoch == 0) and idx in [0, total_steps]:
                self.writer.add_figure(f"train idx {idx}", self.draw_trained_trajectories(pred, gt_vec), epoch + 1)

        if self.reduction == "sum":
            avg_losses = [i / len(self.train_dataloader) / pred.shape[-1] for i in running_losses]
        else:
            avg_losses = [i / len(self.train_dataloader) for i in running_losses]
        current = epoch + 1

        # log losses, gradients and parameters
        for i, loss in enumerate(avg_losses):
            if i == 0:
                self.writer.add_scalar("Loss/train Avg", loss, current)
            else:
                self.writer.add_scalar(f"Loss/train {self.loss_fns[i-1].name()}", loss, current)
        pass
        for tag, value in self.model.named_parameters():
            self.writer.add_histogram(f"Param/{tag}", value, current)
            self.writer.add_scalar(f"Param_norm/{tag}", value.detach().norm().item(), current)
            self.writer.add_scalar(f"Param_mean/{tag}", value.detach().mean().item(), current)
        self.writer.flush()
        return avg_losses[0]

    # Test the model
    def test(self, epoch):
        """Test routine for the model"""
        self.model.eval()
        running_losses = [0.0] * (len(self.loss_fns) + 1)
        if len(self.test_dataloader.dataset) == 0:
            return np.inf
        with torch.no_grad():
            last_idx = len(self.test_dataloader) - 1
            for idx, (input_vec, gt_data) in enumerate(self.test_dataloader):
                # get data
                input_vec = input_vec[..., self.dataset_input_idxs].to(self.device)
                gt_vec = gt_data[..., self.dataset_output_idxs].to(self.device)

                # prediction step
                pred = self.model(input_vec)

                # calculate losses
                losses = [loss_fn(pred, gt_vec) for loss_fn in self.loss_fns]
                total_loss = sum(losses)
                running_losses[0] += total_loss.item()
                for i, loss in enumerate(losses):
                    running_losses[i + 1] += loss.item()

                # Visualize current predictions
                if ((epoch + 1) % self.draw_trajectories == 0 or epoch == 0) and idx in [0, last_idx]:
                    self.writer.add_figure(f"test idx {idx}", self.draw_trained_trajectories(pred, gt_vec), epoch + 1)

            if self.reduction == "sum":
                avg_losses = [i / len(self.test_dataloader) / pred.shape[-1] for i in running_losses]
            else:
                avg_losses = [i / len(self.test_dataloader) for i in running_losses]
            current = epoch + 1

            for i, loss in enumerate(avg_losses):
                if i == 0:
                    self.writer.add_scalar("Loss/test Avg", loss, current)
                else:
                    self.writer.add_scalar(f"Loss/test {self.loss_fns[i-1].name()}", loss, current)
            self.writer.flush()
        return avg_losses[0]

    def run_training(self, epochs=30):
        """Run the complete training routine"""
        test_loss = np.inf
        train_loss = np.inf
        epoch_train_loss = np.inf
        epoch_test_loss = np.inf
        pbar = tqdm.tqdm(
            range(epochs),
            position=0,
            leave=False,
            desc="Epochs",
            ncols=100,
            postfix={"train_loss": train_loss, "test_loss": epoch_test_loss},
        )
        for epoch in pbar:
            epoch_train_loss = self.train(epoch)
            train_loss = epoch_train_loss
            pbar.set_postfix({"train_loss": epoch_train_loss, "test_loss": epoch_test_loss})
            if (epoch + 1) % 10 == 0:
                epoch_test_loss = self.test(epoch)

                if epoch_test_loss <= test_loss:
                    test_loss = epoch_test_loss
                    name = "current_best"
                    self.save_model(name)

        self.writer.close()
        print(f"Training finished with train loss: {train_loss} and test loss: {test_loss}")

    def save_model(self, name: str):
        path = f"{self.model_path}/{name}.pth"
        torch.save(self.model.state_dict(), path)

    def draw_trained_trajectories(self, pred_trajs_gpu, gt_trajs_gpu, draw_centers=False):
        """Draw max 500 predicted and ground truth trajectories, images are send to tensorboard"""
        pos = all([i in self.model.output_labels for i in ["x", "y"]])
        heading = "psi" in self.model.output_labels
        vel = "v" in self.model.output_labels
        acc = "acc" in self.model.output_labels
        gt_trajs = gt_trajs_gpu.cpu().detach().numpy()
        pred_trajs = pred_trajs_gpu.cpu().detach().numpy()
        if draw_centers:
            try:
                centers = self.model.rbflayer.centers.detach().cpu().numpy()
            except AttributeError:
                centers = self.model.weights.detach().cpu().numpy()
        else:
            centers = None
        npics = pos + heading + vel + acc
        nrows = (npics + 1) // 2
        fig, axes = plt.subplots(nrows=nrows, ncols=1 if npics < 2 else 2, figsize=(20, 5 * npics))
        try:
            axes = axes.flatten()
        except AttributeError:
            axes = [axes]
        i = 0
        if pos:
            ax = axes[i]
            i += 1
            idx_x = self.model.output_labels.index("x")
            idx_y = self.model.output_labels.index("y")
            ax.plot(gt_trajs[0, :, idx_x], gt_trajs[0, :, idx_y], "b", label="Ground Truth")
            ax.plot(pred_trajs[0, :, idx_x], pred_trajs[0, :, idx_y], "r", label="Prediction")
            for idx in range(1, min(gt_trajs.shape[0], 500)):
                ax.plot(gt_trajs[idx, :, idx_x], gt_trajs[idx, :, idx_y], "b")
                ax.plot(pred_trajs[idx, :, idx_x], pred_trajs[idx, :, idx_y], color="r")
            if centers is not None:
                ax.plot(centers[:, idx_x], centers[:, idx_y], "gx", markersize=5, label="RBF Centers")
            ax.legend()
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
        if heading:
            try:
                ax = axes[i]
                i += 1
            except TypeError:
                ax = axes
            idx_p = self.model.output_labels.index("psi")
            ax.plot(gt_trajs[0, :, idx_p], "b", label="Ground Truth")
            ax.plot(pred_trajs[0, :, idx_p], color="r", label="Prediction")
            for idx in range(1, min(gt_trajs.shape[0], 500)):
                ax.plot(gt_trajs[idx, :, idx_p], "b")
                ax.plot(pred_trajs[idx, :, idx_p], color="r")
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Orientation")
        if vel:
            try:
                ax = axes[i]
                i += 1
            except TypeError:
                ax = axes
            idx_v = self.model.output_labels.index("v")
            ax.plot(gt_trajs[0, :, idx_v], "b", label="Ground Truth")
            ax.plot(pred_trajs[0, :, idx_v], color="r", label="Prediction")
            for idx in range(1, min(gt_trajs.shape[0], 200)):
                ax.plot(gt_trajs[idx, :, idx_v], "b")
                ax.plot(pred_trajs[idx, :, idx_v], color="r")
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Velocity")
        if acc:
            try:
                ax = axes[i]
                i += 1
            except TypeError:
                ax = axes
            idx_a = self.model.output_labels.index("acc")
            ax.plot(gt_trajs[0, :, idx_a], "b", label="Ground Truth")
            ax.plot(pred_trajs[0, :, idx_a], color="r", label="Prediction")
            for idx in range(1, min(gt_trajs.shape[0], 200)):
                ax.plot(gt_trajs[idx, :, idx_a], "b")
                ax.plot(pred_trajs[idx, :, idx_a], color="r")
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Acceleration")

        return fig

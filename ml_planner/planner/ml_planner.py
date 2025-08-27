__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib.path import Path as GeoPath

from bokeh.models import (
    ColumnDataSource,
    Scatter,
    MultiLine,
    MultiPolygons,
    Ellipse,
    Line,
    Rect,
    HoverTool,
    CDSView,
    DataTable,
    TableColumn,
    IndexFilter,
    BooleanFilter,
    CustomJS,
    Toggle,
    Button,
    HTMLTemplateFormatter,
    CheckboxGroup,
    RangeSlider,
    NumericInput,
)
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row, column, Spacer, gridplot
from bokeh.transform import linear_cmap
from bokeh.palettes import RdYlGn11


from ml_planner.general_utils.data_types import StateTensor, DTYPE
from ml_planner.general_utils.logging import logger_initialization
from ml_planner.general_utils.vehicle_models import VEHICLE_PARAMS

from ml_planner.sampling.sampling_network_utils import load_model

from .cost_functions import Costs
from .planner_utils import (
    convert_globals_to_locals,
    convert_locals_to_globals,
)


class MLPlanner(nn.Module):
    def __init__(
        self,
        sampling_model_path,
        sampling_model_name,
        sampling_model_type,
        log_path,
        logging_level,
        visual_debug_mode,
        sampling_config,
        cost_weights,
    ):
        super().__init__()
        # initialize logger
        self.log_path = log_path
        self.msg_logger = logger_initialization(log_path=log_path, logger="ML_Planner", loglevel=logging_level.upper())
        self.visual_debug_mode = visual_debug_mode

        self.timestep = -1

        # get device
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.msg_logger.info(f"Using {self.device} device")
        # load and initialize sampling model
        sampling_model_path = Path(sampling_model_path) / sampling_model_name
        (
            sampling_model,
            self.model_kwargs,
        ) = load_model(sampling_model_path, sampling_model_type, self.device)
        self.sampling_model = sampling_model.to(self.device)
        self.output_labels = self.model_kwargs["output_labels"]
        self.input_labels = self.model_kwargs["input_labels"]
        self.msg_logger.info(self.sampling_model)

        self.horizon = self.model_kwargs["time"]["planning_horizon"]
        self.dt = self.model_kwargs["time"]["dt"]
        self.num_points_per_traj = self.model_kwargs["num_points_per_traj"]

        # load and initialize cost
        cost_function = Costs(cost_weights, self.device)
        self.cost_function = cost_function.to(self.device)

        # get vehicle parameters
        self.vehicle_dict = self.model_kwargs["vehicle"]
        self.vehicle_params = VEHICLE_PARAMS[self.vehicle_dict["vehicle_type"]]
        self.vehicle_params.longitudinal.v_max = self.model_kwargs["bounds"][self.output_labels.index("v")][1]
        self.ego_shape = [self.vehicle_params.l, self.vehicle_params.w]
        self.v_switch = self.vehicle_params.longitudinal.v_switch
        self.a_max = self.vehicle_params.longitudinal.a_max
        self.v_max = self.vehicle_params.longitudinal.v_max

        # initialize variables
        self.ref_path_local: StateTensor = None
        self.current_state_global: StateTensor = None
        self.current_state_local: StateTensor = None
        self.obs_preds_local: list[dict] = None
        self.lane_preds_local: list[dict] = None
        self.desired_velocity: float = None

        self.all_trajs_local = None
        self.optimal_traj_global = None
        self.optimal_traj_local = None
        self.optimal_costs_sum = torch.inf
        self.optimal_costs = self.traj_costs_sum = self.traj_costs = None
        self.x0_loc = None

        # set sampling configurations for final sampling nodes
        sampling_model_params = self.model_kwargs["sampling"]
        self.sample_feasible_region = sampling_config["sample_feasible_region"]
        self.resample_every_n_steps = sampling_config["latest_resampling_every_n_steps"]
        self.use_improved_trajectories = sampling_config["use_improved_trajectories"]
        self.resampling_counter = -1
        # make sure config values are in trained area
        # x range (calculated in each iteration based on current velocity)
        self.dx = sampling_config["dx"] if sampling_config["dx"] else sampling_model_params["dx"]
        x_model_limits = self.model_kwargs["bounds"][self.output_labels.index("x")]
        x_min = max(sampling_config["x_min"], x_model_limits[0]) if sampling_config["x_min"] else x_model_limits[0]
        x_max = min(sampling_config["x_max"], x_model_limits[1]) if sampling_config["x_max"] else x_model_limits[1]
        print(self.dx)
        self.x_limits = (x_min, x_max)
        # y range (calculated in each iteration based on current velocity)
        self.dy = sampling_config["dy"] if sampling_config["dy"] else sampling_model_params["dy"]
        y_model_limits = self.model_kwargs["bounds"][self.output_labels.index("y")]
        y_min = max(sampling_config["y_min"], y_model_limits[0]) if sampling_config["y_min"] else y_model_limits[0]
        y_max = min(sampling_config["y_max"], y_model_limits[1]) if sampling_config["y_max"] else y_model_limits[1]
        self.y_limits = (y_min, y_max)
        # psi range
        dpsi = (
            max(sampling_config["dpsi"], sampling_model_params["dpsi"])
            if sampling_config["dpsi"]
            else sampling_model_params["dpsi"]
        )
        psi_model_limits = self.model_kwargs["bounds"][self.output_labels.index("psi")]
        psi_min = (
            max(sampling_config["psi_min"], psi_model_limits[0]) if sampling_config["psi_min"] else psi_model_limits[0]
        )
        psi_max = (
            min(sampling_config["psi_max"], psi_model_limits[1]) if sampling_config["psi_max"] else psi_model_limits[1]
        )
        num_psi = int((psi_max - psi_min) / dpsi) + 1
        self.psi_range = torch.linspace(psi_min, psi_max, num_psi, device=self.device, dtype=DTYPE)

    def extra_repr(self) -> str:
        return f"""ML_Planner Module"""

    @property
    def all_trajs_global(self):
        return convert_locals_to_globals(self.all_trajs_local, self.current_state_global)

    def plan(
        self,
        current_state_global: StateTensor,
        desired_velocity: float,
        reference_path_local: StateTensor,
        obstacle_predictions_local: list[dict],
        lane_predictions_local: list[dict],
    ):
        assert current_state_global.state_variables == self.output_labels, "State labels do not match model labels"
        """calculate optimal trajectory for current state"""
        self.current_state_global = current_state_global
        self.current_state_local = convert_globals_to_locals(current_state_global, current_state_global)
        self.desired_velocity = desired_velocity
        self.ref_path_local = reference_path_local
        self.obs_preds_local = obstacle_predictions_local
        self.lane_preds_local = lane_predictions_local

        self.timestep += 1
        self.resampling_counter += 1
        self.msg_logger.debug("Calculate Trajectory")
        self.sampling_model.eval()
        with torch.no_grad():
            # sample trajectories in local reference frame
            self.all_trajs_local = self._sample_trajectories()

            self.msg_logger.debug(f"Sampled {self.all_trajs_local.num_batches} trajectories")
            # calculate costs
            traj_costs_sum, traj_costs = self.cost_function(
                self.all_trajs_local,
                self.ego_shape,
                self.ref_path_local,
                self.desired_velocity,
                self.obs_preds_local,
                self.lane_preds_local,
            )
            self.msg_logger.debug(f"Calculated costs for {self.all_trajs_local.num_batches} trajectories")

            self.traj_costs_sum, cost_idxs_sorted = traj_costs_sum.sort()
            self.traj_costs = traj_costs[cost_idxs_sorted]
            self.all_trajs_local.sort_indices(cost_idxs_sorted)

            min_costs = self.traj_costs_sum[0]

        resampling = self.resampling_counter % self.resample_every_n_steps == 0
        improved = min_costs < self.optimal_costs_sum and self.use_improved_trajectories
        if resampling or improved:
            # if resampling or improved update optimal trajectory
            self.msg_logger.info("New trajectory" if resampling else "Found improved trajectory")
            self.optimal_costs_sum = min_costs
            self.optimal_costs = self.traj_costs[0]
            self.optimal_traj_local = StateTensor(
                states=self.all_trajs_local.states[0], covs=self.all_trajs_local.covs[0], device=self.device
            )

            self.msg_logger.info(f"Optimal Trajectory with costs {self.optimal_costs_sum.cpu().detach().numpy()} ")
            self.msg_logger.debug(f"Cost weights: {self.cost_function.cost_terms} ")
            self.msg_logger.debug(f"wo weight: {self.optimal_costs.cpu().detach().numpy()} ")
            self.msg_logger.debug(
                f"""w weights: {self.optimal_costs.cpu().detach().numpy() *
                                  self.cost_function.cost_weights.cpu().detach().numpy()}"""
            )

        else:
            self.msg_logger.info("Using previously calculated optimal trajectory")
            self.optimal_traj_local = StateTensor(
                states=self.optimal_traj_local.states[1:], covs=self.optimal_traj_local.covs[1:], device=self.device
            )

        # transform to global reference frame
        self.optimal_traj_global = convert_locals_to_globals(self.optimal_traj_local, self.current_state_global)

        if self.visual_debug_mode:
            self.render_current_state(
                local_coosy=False, title=f"global_frame_timestep_{self.timestep}", show_fig=True, plot_scatter=True
            )
        return self.optimal_traj_global

    def _sample_trajectories(self):
        """sample trajectories in local reference frame"""
        assert [
            i[:-2] for i in self.sampling_model.input_labels[: self.current_state_local.num_variables]
        ] == self.current_state_local.state_variables, "Input labels do not match current state labels"

        v0 = self.current_state_local.v
        # Convert CUDA tensor to CPU for numpy operations
        v0_cpu = v0.detach().cpu().numpy()

        # x range
        def _forward_pos(v0):
            # forward simulation of vehicle dynamics to get max position
            x = np.zeros(self.num_points_per_traj)
            v = np.zeros(self.num_points_per_traj)
            v[0] = v0
            for i in range(1, self.num_points_per_traj):

                def acc(v):
                    # constraint acceleration
                    # Convert v0 to tensor for consistent operations
                    v0_tensor = torch.tensor(v0, dtype=DTYPE, device=self.device)
                    a = self.a_max * torch.min(torch.tensor(1, dtype=DTYPE, device=self.device), self.v_switch / v0_tensor) if v0 < self.v_max else 0
                    return a
                # euler method
                v[i] = v[i - 1] + acc(v[i - 1]) * self.dt
                x[i] = x[i - 1] + v[i - 1] * self.dt

                # RK4 method
                # vi = v[i - 1]
                # xi = x[i - 1]

                # RK4 for v
                # k1v = acc(vi)
                # k2v = acc(vi + 0.5 * dt * k1v)
                # k3v = acc(vi + 0.5 * dt * k2v)
                # k4v = acc(vi + dt * k3v)
                # v[i] = vi + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4x)
                # RK4 for x
                # k1x = vi
                # k2x = vi + 0.5 * dt * k1v
                # k3x = vi + 0.5 * dt * k2v
                # k4x = vi + dt * k3v
                # x[i] = xi + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            return x, v

        x, _ = _forward_pos(v0_cpu)
        x_min = max(np.ceil(v0_cpu**2 / (-2 * self.a_max)), self.x_limits[0])
        x_max = min(
            np.floor(x[-1]), self.x_limits[1]
        )
        num_x = int((x_max - x_min) / self.dx) + 1
        x_range = torch.linspace(x_min, x_max, num_x, device=self.device, dtype=DTYPE)

        # y range
        y_max = min(x_max, self.y_limits[1])
        y_min = max(-x_max, self.y_limits[0])
        num_y = int((y_max - y_min) / self.dy) + 1
        y_range = torch.linspace(y_min, y_max, num_y, device=self.device, dtype=DTYPE)
        xy_loc = torch.cartesian_prod(x_range, y_range)

        if self.sample_feasible_region:
            # sample within driveable region, avoid obstacles etc.
            # create a mask for all points that are within the road boundary
            boundary = [i["states"] for i in self.lane_preds_local if i["boundary"]][0]
            # Convert CUDA tensor to CPU for matplotlib
            boundary_coords_cpu = boundary.coords.detach().cpu().numpy()
            boundary_path = GeoPath(boundary_coords_cpu)
            # Convert xy_loc to CPU for matplotlib operations
            xy_loc_cpu = xy_loc.detach().cpu().numpy()
            mask = boundary_path.contains_points(xy_loc_cpu)
            xy_loc = xy_loc[mask]

        psi_range = self.psi_range.repeat(xy_loc.shape[0])

        xy_loc = xy_loc.repeat_interleave(self.psi_range.shape[0], dim=0)
        xF = torch.cat((xy_loc, psi_range.unsqueeze(1)), dim=1)

        x0_loc = self.current_state_local.states.unsqueeze(0).expand(xF.shape[0], -1)
        input_x = torch.hstack([x0_loc, xF]).to(self.device)
        # optain trajectories
        pred = self.sampling_model(input_x)

        if self.sample_feasible_region:
            # sample within driveable region
            # create a mask for all points that are within the road boundary
            # Convert pred to CPU for matplotlib operations
            pred_cpu = pred.detach().cpu().numpy()
            mask = [all(boundary_path.contains_points(pred_cpu[i, :, :2])) for i in range(pred_cpu.shape[0])]
            pred = pred[mask]

        mask = torch.all(pred[:, 1:, 0] >= pred[:, :-1, 0], dim=1)
        pred = pred[mask]

        trajs = StateTensor(states=pred, device=self.device)
        covs = self.current_state_local.covs.repeat(trajs.num_batches, trajs.num_states, 1, 1)
        trajs.add_cov(covs)
        return trajs

    def render_current_state(self, local_coosy=True, title="Debug_ML_Planner", show_fig=False, plot_scatter=False):
        render_dir = self.log_path + "/planner_plots"
        os.makedirs(render_dir, exist_ok=True)
        output_file(filename=render_dir + "/" + title.lower() + ".html", title=title)

        fig = self._create_figure(local_coosy, title, plot_scatter)
        # checkboxes
        checkboxes = self._create_checkboxes(fig)
        # plot buttons
        buttons = self._create_plot_buttons(fig, checkboxes)

        # trajectory details
        data_table = self._create_data_table(fig)
        traj_buttons = self._create_traj_buttons(fig, data_table, plot_scatter=plot_scatter)
        traj_figures = self._create_trajectory_debugger(fig)

        plot_layout = row(
            column(
                Spacer(height=10, sizing_mode="scale_width"),
                row(Spacer(width=60, sizing_mode="scale_width"), *buttons, sizing_mode="scale_width"),
                row(
                    fig,
                    sizing_mode="scale_width",
                ),
                sizing_mode="scale_both",
            ),
            column(
                Spacer(height=10, sizing_mode="scale_width"),
                row(*traj_buttons, sizing_mode="scale_width"),
                row(checkboxes, data_table, sizing_mode="scale_width"),
                gridplot(traj_figures, ncols=2, sizing_mode="scale_width", toolbar_location="right"),
            ),
            Spacer(width=30),
            sizing_mode="scale_both",
        )

        if show_fig:
            show(plot_layout)
            pass
        save(plot_layout)

    def _create_figure(self, local_coosy=True, title="Debug_ML_Planner", plot_scatter=False):
        # all trajectories
        trajs = self.all_trajs_local.detach() if local_coosy else self.all_trajs_global.detach()
        ego_state = self.current_state_local.detach() if local_coosy else self.current_state_global.detach()
        total_costs = self.traj_costs_sum.detach()
        traj_costs = self.traj_costs.detach()
        traj_costs_weighted = traj_costs * self.cost_function.cost_weights.detach()
        # optimal trajectory
        opt_traj = self.optimal_traj_local.detach() if local_coosy else self.optimal_traj_global.detach()
        # reference path
        ref_path = self.ref_path_local.detach()
        boundary = [i for i in self.lane_preds_local if i["boundary"] == True][0]["states"].detach()

        # predictions
        obs_preds = self.obs_preds_local
        lane_preds = self.lane_preds_local

        if not local_coosy:
            ref_path = convert_locals_to_globals(self.ref_path_local, self.current_state_global)
            boundary = convert_locals_to_globals(boundary, self.current_state_global)
            for pred in obs_preds:
                pred["states"] = convert_locals_to_globals(pred["states"], self.current_state_global)
            for pred in lane_preds:
                pred["states"] = convert_locals_to_globals(pred["states"], self.current_state_global)

        min_val = (trajs.coords.amin(dim=(0, 1)) - 5).tolist()
        delta = (trajs.coords.amax(dim=(0, 1)) - trajs.coords.amin(dim=(0, 1))).max().item() + 10

        x_range = [min_val[0], min_val[0] + delta]
        y_range = [min_val[1], min_val[1] + delta]
        tools = ["pan", "wheel_zoom", "box_zoom", "tap", "save", "reset"]
        fig = figure(
            x_range=x_range,
            y_range=y_range,
            width=800,
            height=800,
            title=title,
            name=title,
            tools=tools,
            match_aspect=True,
            active_scroll="wheel_zoom",
            margin=[10, 10, 10, 30],
            output_backend="canvas",
            min_width=400,
            max_width=1100,
            min_height=400,
            max_height=1100,
            sizing_mode="scale_both",
        )
        fig.axis.visible = True
        fig.xgrid.visible = False
        fig.ygrid.visible = False

        # draw road boundary
        name = "road boundary"
        x, y = boundary.coords.T.tolist()
        data_source_road = ColumnDataSource(
            data=dict(
                xs=[[[x]]],
                ys=[[[y]]],
            ),
            name=name,
            tags=["data"],
        )
        glyph_road = MultiPolygons(
            xs="xs",
            ys="ys",
            line_color="black",
            line_width=1,
            fill_color="#b5b5b5",
            fill_alpha=0.5,
            name=name,
            tags=["glyph"],
        )
        fig.add_glyph(data_source_road, glyph_road, name=name, tags=["renderer"])

        # draw reference path
        name = "reference path"
        x, y = ref_path.coords.T.tolist()
        data_source_ref_path = ColumnDataSource(
            data=dict(
                x=x,
                y=y,
            ),
            name=name,
            tags=["data"],
        )
        glyph_ref_path = Line(x="x", y="y", line_color="green", line_width=1, name=name, tags=["glyph"])
        fig.add_glyph(data_source_ref_path, glyph_ref_path, name=name, tags=["renderer"])

        # draw predictions (if available)
        def _add_pred(preds, name):
            mean_x = []
            mean_y = []
            pred_w = []
            pred_h = []
            angles = []
            pred_col = []
            pred_nstd = []
            gt_type = []
            gt_id = []
            for pred in preds:
                states = pred["states"]
                x, y = states.coords.T.tolist()

                ew, ev = torch.linalg.eigh(states.coords_cov)
                angle = torch.atan2(ev[:, 1, 0], ev[:, 0, 0])

                for n_std, color in zip([1.0, 0.5, 0.2], ["yellow", "orange", "red"]):
                    w = (
                        torch.sqrt(ew[:, 0]) * n_std
                    )
                    h = (
                        torch.sqrt(ew[:, 1]) * n_std
                    )
                    mean_x += x
                    mean_y += y
                    pred_w += w
                    pred_h += h

                    angles += angle
                    pred_col += [color] * len(x)
                    pred_nstd += [n_std] * len(x)
                    gt_type += list(pred["types"].keys()) * len(x)
                    gt_id += [pred["id"]] * len(x)

            data_source_pred = ColumnDataSource(
                data=dict(
                    x=mean_x,
                    y=mean_y,
                    w=pred_w,
                    h=pred_h,
                    color=pred_col,
                    angle=angles,
                    nstd=pred_nstd,
                    gt_type=gt_type,
                    gt_id=gt_id,
                ),
                name=name,
                tags=["data"],
            )
            glyph_pred = Ellipse(
                x="x",
                y="y",
                width="w",
                height="h",
                line_alpha=0,
                fill_color="color",
                fill_alpha=0.4,
                angle="angle",
                angle_units="rad",
                name=name,
                tags=["glyph"],
            )
            renderer = fig.add_glyph(data_source_pred, glyph_pred, name=name, tags=["renderer"])
            tooltips = [
                ("detection", "@gt_id"),
                ("type", "@gt_type"),
                ("nstd", "@nstd"),
                ("x [m]", "$x"),
                ("y [m]", "$y"),
            ]
            hover = HoverTool(
                renderers=[renderer],
                tooltips=tooltips,
            )

            fig.add_tools(hover)
            return

        _add_pred(obs_preds, name="obs predictions")
        _add_pred(lane_preds, name="lane predictions")

        # GT Preditions at t_plan=0
        gt_x = []
        gt_y = []
        gt_w = []
        gt_h = []
        gt_psi = []
        gt_v = []
        gt_type = []
        gt_id = []
        for pred in obs_preds:
            states = pred["states"]

            [x, y, psi, v] = states.states[0].tolist()
            gt_x.append(x)
            gt_y.append(y)
            gt_w.append(pred["shape"]["length"])
            gt_h.append(pred["shape"]["width"])
            gt_psi.append(np.round(psi, 3))
            gt_v.append(v)
            gt_type.append(list(pred["types"].keys()))
            gt_id.append(pred["id"])

        name = "gt_predictions"
        data_source_gt = ColumnDataSource(
            data=dict(
                x=gt_x,
                y=gt_y,
                w=gt_w,
                h=gt_h,
                v=gt_v,
                angle=gt_psi,
                gt_type=gt_type,
                gt_id=gt_id,
            ),
            name=name,
            tags=["data"],
        )

        glyph_gt = Rect(
            x="x",
            y="y",
            width="w",
            height="h",
            angle="angle",
            fill_color="blue",
            line_color="black",
            fill_alpha=0.5,
            name=name,
            tags=["glyph"],
        )
        renderer = fig.add_glyph(data_source_gt, glyph_gt, name=name, tags=["renderer"])

        tooltips = [
            ("detection", "@gt_id"),
            ("type", "@gt_type"),
            ("x0 [m]", "@x"),
            ("y0 [m]", "@y"),
            ("psi0 [rad]", "@angle"),
            ("v0 [m/s]", "@v"),
        ]
        hover = HoverTool(
            renderers=[renderer],
            tooltips=tooltips,
        )

        fig.add_tools(hover)

        # draw ego vehicle
        name = "ego"
        state0 = ego_state.states.tolist()
        wb_rear_axle = self.vehicle_params.b
        data_source_ego = ColumnDataSource(
            data=dict(
                x=[state0[ego_state.x_idx] + wb_rear_axle * np.cos(state0[ego_state.psi_idx])],
                y=[state0[ego_state.y_idx] + wb_rear_axle * np.sin(state0[ego_state.psi_idx])],
                x_rear=[state0[ego_state.x_idx]],
                y_rear=[state0[ego_state.y_idx]],
                w=[self.vehicle_params.l],
                h=[self.vehicle_params.w],
                angle=np.round([state0[ego_state.psi_idx]], 3),
                v=[state0[ego_state.v_idx]],
            ),
            name=name,
            tags=["data"],
        )
        glyph_ego = Rect(
            x="x",
            y="y",
            width="w",
            height="h",
            angle="angle",
            fill_color="orange",
            line_color="black",
            fill_alpha=0.5,
            name=name,
            tags=["glyph"],
        )
        renderer = fig.add_glyph(data_source_ego, glyph_ego, name=name, tags=["renderer"])
        tooltips = [
            ("object", "ego vehicle"),
            ("x_rear [m]", "@x_rear"),
            ("y_rear [m]", "@y_rear"),
            ("psi [rad]", "@angle"),
            ("v [m/s]", "@v"),
        ]
        hover = HoverTool(
            renderers=[renderer],
            tooltips=tooltips,
        )

        fig.add_tools(hover)

        # draw all trajectories
        name = "all trajs"
        states = trajs.states.flip(dims=[0]).permute(2, 0, 1)
        x, y, psi, v, delta = states.tolist()
        xf, yf, psif, vf, deltaf = states[..., -1].round(decimals=3).tolist()
        data_source_traj = ColumnDataSource(
            data=dict(
                x=x,
                xf=xf,
                y=y,
                yf=yf,
                v=v,
                vf=vf,
                psi=psi,
                psif=psif,
                cost=np.round(total_costs.flip(dims=[0]).tolist(), 3),
                T=[np.arange(0, trajs.num_states)] * trajs.num_batches,
                colormapper=np.round(total_costs.flip(dims=[0]).tolist(), 3),
                line_width=[1] * trajs.num_batches,
                line_alpha=[0.5] * trajs.num_batches,
                index=[trajs.num_batches - i for i in range(trajs.num_batches)],
                selected=[False] * trajs.num_batches,
                row_color=[""] * trajs.num_batches,
            ),
            name=name,
            tags=["data"],
        )
        costs = np.round(traj_costs.flip(dims=[0]), 4)
        costs_w = np.round(traj_costs_weighted.flip(dims=[0]), 4)
        data_source_traj.data.update(
            {
                key
                + "_(unweighted)": [
                    f"{np.round(costs_w[idx, c_idx].item(),3)}      ({np.round(costs[idx, c_idx].item(),3)})"
                    for idx in range(costs.shape[0])
                ]
                for c_idx, key in enumerate(self.cost_function.cost_terms)
            },
        )

        mapper = linear_cmap(
            field_name="colormapper",
            palette=RdYlGn11,
            low=float(min(total_costs)),
            high=float(max(total_costs)),
            nan_color="#00ffff",
            low_color="red",
        )
        glyph_traj = MultiLine(
            xs="x",
            ys="y",
            line_color=mapper,
            line_width="line_width",
            line_alpha="line_alpha",
            name=name,
            tags=["glyph"],
        )

        renderer = fig.add_glyph(data_source_traj, glyph_traj, name=name, tags=["renderer"])
        tooltips = [
            ("object", "trajectory (@index)"),
            ("cost", "@cost"),
            ("y (goal) [m]", "$y (@yf)"),
            ("x (goal) [m]", "$x (@xf)"),
            ("psi goal [rad]", "@psif"),
            ("v goal [m/s]", "@vf"),
        ]
        hover = HoverTool(
            renderers=[renderer],
            tooltips=tooltips,
        )
        fig.add_tools(hover)

        # add final traj points
        # Add scatter glyph for all final x, y positions of the trajectories
        name = "final traj points"

        data_source_final_points = ColumnDataSource(
            data=dict(
                x=xf,
                y=yf,
            ),
            name=name,
            tags=["data"],
        )
        glyph_final_points = Scatter(
            x="x",
            y="y",
            size=8,
            marker="x",
            fill_color="blue",
            line_color="black",
            fill_alpha=0.8,
            name=name,
            tags=["glyph"],
        )
        renderer = fig.add_glyph(data_source_final_points, glyph_final_points, name=name, tags=["renderer"])

        if plot_scatter:
            # add discretization points of trajs
            name = "all trajs disc"
            x = [i for xx in x for i in xx]
            y = [i for yy in y for i in yy]
            data_source_traj = ColumnDataSource(
                data=dict(
                    x=x,
                    y=y,
                ),
                name=name,
                tags=["data"],
            )
            glyph_traj = Scatter(
                x="x",
                y="y",
                size=3,
                marker="circle_x",
                name=name,
                tags=["glyph"],
            )
            rend = fig.add_glyph(data_source_traj, glyph_traj, name=name, tags=["renderer"])
            rend.visible = False

        # draw optimal trajectory
        name = "optimal trajectory"
        x, y, psi, v, delta = opt_traj.states.T.tolist()
        data_source_opt = ColumnDataSource(
            data=dict(
                xs=x,
                xf=[np.round(x[-1], 3)] * opt_traj.num_states,
                ys=y,
                yf=[np.round(y[-1], 3)] * opt_traj.num_states,
                v=v,
                vf=[np.round(v[-1], 3)] * opt_traj.num_states,
                psi=psi,
                psif=[np.round(psi[-1], 3)] * opt_traj.num_states,
                cost=[np.round(self.optimal_costs_sum.detach(), 3)] * opt_traj.num_states,
            ),
            name=name,
            tags=["data"],
        )

        glyph_opt = Line(x="xs", y="ys", line_color="black", line_width=3, name=name, tags=["glyph"])
        renderer = fig.add_glyph(data_source_opt, glyph_opt, name=name, tags=["renderer"])
        tooltips = [
            ("object", "optimal trajectory"),
            ("cost", "@cost"),
            ("y (goal) [m]", "@ys (@yf)"),
            ("x (goal) [m]", "@xs (@xf)"),
            ("psi (goal) [rad]", "@psi (@psif)"),
            ("v (goal) [m/s]", "@v (@vf)"),
        ]
        hover = HoverTool(
            renderers=[renderer],
            tooltips=tooltips,
        )
        fig.add_tools(hover)
        return fig

    @staticmethod
    def _create_plot_buttons(figure, checkbox_group):
        buttons = []

        # general purpose buttons
        def _create_button(label, checkbox_group, input_index):
            button = Button(
                label=label,
                button_type="primary",
                sizing_mode="stretch_width",
                width=150,
                min_width=75,
                max_width=200,
                margin=[15, 10, 0, 10],
            )
            button.js_on_click(
                CustomJS(
                    args=dict(checkbox_group=checkbox_group, input_index=input_index),
                    code="""
                            checkbox_group.active = input_index;""",
                )
            )
            return button

        # show all
        button_select = _create_button("Show Everything", checkbox_group, list(range(len(checkbox_group.labels))))
        buttons.append(button_select)
        # hide all
        button_unselect = _create_button("Empty Plot", checkbox_group, [])
        buttons.append(button_unselect)
        return buttons

    @staticmethod
    def _create_checkboxes(figure):
        title = [rend.name for rend in figure.renderers]
        active = list(range(len(title)))
        try:
            active.remove(title.index("all trajs disc"))
        except ValueError:
            pass

        checkbox_group = CheckboxGroup(
            labels=title,
            active=active,
            margin=[35, 5, 5, 5],
        )
        checkbox_group.js_on_change(
            "active",
            CustomJS(
                args=dict(checkbox_group=checkbox_group, figure=figure),
                code="""
            for (var i = 0; i < figure.renderers.length; i++) {
                figure.renderers[i].visible = checkbox_group.active.includes(i);
            }
        """,
            ),
        )
        return checkbox_group

    def _create_data_table(self, figure):
        cds = figure.select(name="all trajs", tags=["data"])[0]
        # create data table
        dic = dict(index=[], cost=[])

        dic.update({key + "_(unweighted)": [] for key in self.cost_function.cost_terms})

        template = """
            <div title="<%= x %>" style="font-size: 500px;">
            </div>
            <div style="background:<%= row_color %>;">
            <%= value %>
            </div>
            """

        formatter = HTMLTemplateFormatter(template=template)

        columns = [
            TableColumn(
                field=key,
                title=f"{key.replace('_', ' ').title()}",  # .replace(' (', '<br>(')}",
                formatter=formatter,
                width=max((len(key) - 12) * 6, 55),
            )
            for key in dic
        ]
        margin = [20, 10, 10, 30]

        data_table = DataTable(
            source=cds,
            columns=columns,
            autosize_mode="none",
            frozen_columns=2,
            selectable=True,
            reorderable=False,
            min_height=250,
            max_height=300,
            margin=margin,
            index_position=None,
            sizing_mode="scale_both",
        )
        return data_table

    @staticmethod
    def _create_traj_buttons(figure, data_table, plot_scatter=False):
        buttons = []
        # filter
        cds = figure.select(name="all trajs", tags=["data"])[0]
        all_ind = list(range(len(cds.data["x"])))
        index_filter = IndexFilter(indices=all_ind)
        index_filter_table_selection = IndexFilter(indices=all_ind)
        all_bool = [True for _ in all_ind]
        bool_filter = BooleanFilter(booleans=all_bool)

        view = CDSView(filter=index_filter & bool_filter)

        # add filter to main plot
        main_figure_traj_renderer = figure.select(name="all trajs", tags="renderer")[0]
        main_figure_traj_renderer.view = view
        # for scatter plot:
        if plot_scatter:
            cds_disc = figure.select(name="all trajs disc", tags=["data"])[0]
            all_ind_disc = list(range(len(cds_disc.data["x"])))
            index_filter_disc = IndexFilter(indices=all_ind_disc)
            view_disc = CDSView(filter=index_filter_disc)
            main_figure_traj_disc_renderer = figure.select(name="all trajs disc", tags="renderer")[0]
            main_figure_traj_disc_renderer.view = view_disc
        else:
            all_ind_disc = []
            index_filter_disc = None

        # trajectory slider
        slider = RangeSlider(
            start=0,
            end=len(all_ind) - 1,
            value=(0, len(all_ind) - 1),
            step=1,
            title="Trajectory Index",
            sizing_mode="stretch_width",
            width=250,
            min_width=150,
            max_width=350,
            margin=[10, 15, 0, 15],
        )

        # NumericInput trajectory selector
        low_button = NumericInput(value=0, low=0, high=len(all_ind) - 1, width=55, margin=[10, 5, 10, 5])
        high_button = NumericInput(value=slider.end, low=0, high=len(all_ind) - 1, width=55, margin=[10, 5, 10, 5])

        # range slider callback
        cb = CustomJS(
            args=dict(src=cds, low=low_button, high=high_button, sl=slider),
            code="""
                    if (typeof window.rangeSliderTimeout !== 'undefined') {
                        clearTimeout(window.rangeSliderTimeout);
                    }
                    window.rangeSliderTimeout = setTimeout(function() {
                        low.value = Math.round(sl.value[0]);
                        high.value = Math.round(sl.value[1]);
                        low.change.emit();
                        high.change.emit();
                    }, 200);  // Delay in milliseconds (500ms = 0.5s)
                        """,
        )
        slider.js_on_change("value", cb)

        # numeric input callbacks
        cb_button = CustomJS(
            args=dict(
                sl=slider,
                low=low_button,
                high=high_button,
                src=cds,
                # src2=cds_disc,
                filt_plot=index_filter,
                filt_plot_disc=index_filter_disc,
                filt_table=index_filter_table_selection,
            ),
            code="""
                // Filter the data source to show only trajectories with index > low.value
                var inds = [];
                var inds_disc = [];
                var j = 0;
                var length = src.data['x'].length;
                var num_x = src.data['x'][0].length;

                high.low=Math.round(low.value);
                low.high = Math.round(high.value);

                for (var i = low.value ; i <= high.value; i++) {
                        j = length - i - 1;
                        inds.push(j);
                        for (var k = 0; k < num_x; k++) {
                            inds_disc.push(j*num_x + k);
                        }
                        // inds.push(i);
                    }
                filt_plot.indices = inds;
                filt_table.indices = inds;
                if (filt_plot_disc != null) {
                    filt_plot_disc.indices = inds_disc;
                    }
                inds = [];
                inds_disc = [];

                src.change.emit();
               //  src2.change.emit();
            """,
        )

        low_button.js_on_change("value", cb_button)
        high_button.js_on_change("value", cb_button)

        # update index limits
        update_title_js = CustomJS(
            args=dict(low_button=low_button, high_button=high_button, sl=slider),
            code="""
       //     low_button.title = `Start Index (${low_button.low}...${low_button.high})`;
       //     high_button.title = `End Index (${high_button.low}...${high_button.high})`;

            sl.value = [low_button.value , high_button.value];
            sl.start = low_button.low;
            sl.end = high_button.high;

            low_button.change.emit();
            high_button.change.emit();
        """,
        )

        # Attach the callback to the change event of low and high properties
        low_button.js_on_change("low", update_title_js)
        low_button.js_on_change("high", update_title_js)
        high_button.js_on_change("low", update_title_js)
        high_button.js_on_change("high", update_title_js)
        # table filter
        data_table_view = CDSView(filter=bool_filter & index_filter_table_selection)
        data_table.view = data_table_view
        button_table1 = Toggle(
            label="Table: Selected Trajs Only: Off",
            margin=[15, 10, 20, 10],
            active=False,
            button_type="primary",
            width=150,
            max_width=150,
            min_width=75,
            sizing_mode="stretch_width",
        )
        button_table1.js_on_click(
            CustomJS(
                args=dict(btn=button_table1, src=cds, filt=index_filter_table_selection, all_ind=all_ind),
                code="""
                            // change appearance
                            const indices = src.selected.indices;
                            const data = src.data;
                            if (btn.active) {
                                btn.button_type = "warning";
                                btn.label="Table: Selected Trajs Only: On";
                                if (src.selected.indices.length > 0) {
                                    filt.indices = src.selected.indices;
                                    for (let i=0; i < data['selected'].length; i++) {
                                        data['row_color'][i] = '';
                                        }
                                    }
                                }
                            else {
                                    btn.button_type = "primary";
                                    btn.label="Table: Selected Trajs Only: Off";
                                    filt.indices = all_ind;
                                    for (let i = 0; i < indices.length; i++) {
                                        data['row_color'][indices[i]] = 'yellow'; }
                                    }

                            src.change.emit();
                            """,
            )
        )

        button_table2 = Toggle(
            label="Plots: Selected Trajs Only: Off",
            margin=[15, 10, 20, 10],
            active=False,
            button_type="primary",
            width=150,
            max_width=150,
            min_width=75,
            sizing_mode="stretch_width",
        )
        button_table2.js_on_click(
            CustomJS(
                args=dict(
                    btn=button_table2,
                    src=cds,
                    filt=index_filter,
                    filt_disc=index_filter_disc,
                    all_ind=all_ind,
                    all_ind_disc=all_ind_disc,
                ),
                code="""
                            // change appearance
                            var indices = src.selected.indices;
                            var data = src.data;
                            var num_x = src.data['x'][0].length;
                            var inds_disc = [];
                            if (btn.active) {
                                btn.button_type = "warning";
                                btn.label="Plots: Selected Trajs Only: On";
                                if (src.selected.indices.length > 0) {
                                    filt.indices = src.selected.indices;
                                    // console.log('selected:' + src.selected.indices);
                                    for (let j = 0; j < indices.length; j++) {
                                        for (var k = 0; k < num_x; k++) {
                                            inds_disc.push(src.selected.indices[j]*num_x + k);
                                            }
                                        }
                                    if (filt_disc != null) {
                                        filt_disc.indices = inds_disc;
                                        }
                                    for (let i=0; i < data['selected'].length; i++) {
                                        data['row_color'][i] = '';
                                        }
                                    }
                                }
                            else {
                                    // inds_disc = [];
                                    btn.button_type = "primary";
                                    btn.label="Plots: Selected Trajs Only: Off";
                                    filt.indices = all_ind;
                                    if (filt_disc != null) {
                                        filt_disc.indices = all_ind_disc;
                                        }
                                    for (let i = 0; i < indices.length; i++) {
                                        data['row_color'][indices[i]] = 'yellow'; }
                                }

                            src.change.emit();
                   //         src2.change.emit();
                            """,
            )
        )
        buttons.append(button_table1)
        buttons.append(button_table2)
        buttons.append(slider)
        buttons.append(low_button)
        buttons.append(high_button)

        # CustomJS callback to update the selected line
        callback = CustomJS(
            args=dict(
                source=cds,
                filt_table=index_filter_table_selection,
                filt_plot=index_filter,
                filt_plot_disc=index_filter_disc,
                all_ind=all_ind,
                all_ind_disc=all_ind_disc,
                btn1=button_table1,
                btn2=button_table2,
                table=data_table,
            ),
            code="""
                    var indices = source.selected.indices;
                    var data = source.data;
                    var inds_disc = [];
                    var num_x = data['x'][0].length;
                    // console.log('indices:' +indices);

                    for (let i=0; i < data['selected'].length; i++) {
                                data['row_color'][i] = '';
                            }
                    if (source.selected.indices.length >0 & btn1.active) {
                            filt_table.indices = source.selected.indices;

                        }
                    else {
                        if (btn2.active) {
                            filt_plot.indices = source.selected.indices;
                            if (filt_plot_disc != null) {
                                for (let j = 0; j < source.selected.indices.length; j++) {
                                    for (var k = 0; k < num_x; k++) {
                                        inds_disc.push(source.selected.indices[j]*num_x + k);
                                            }
                                        }
                                filt_plot_disc.indices = inds_disc;
                            }
                        }
                        else {
                            filt_table.indices = all_ind;
                            filt_plot.indices = all_ind;
                            if (filt_plot_disc != null) {
                                filt_plot_disc.indices = all_ind_disc;
                                }
                            // inds_disc = [];
                            for (let i = 0; i < indices.length; i++) {
                                data['row_color'][indices[i]] = 'yellow'; }
                            }
                        }

                    // trajectory plots
                    for (let i = 0; i < data['selected'].length; i++) {
                        data['selected'][i] = 0;
                        data['line_width'][i] = 1;
                        data['line_alpha'][i] = 0.5;
                        data['colormapper'][i] = data['cost'][i];
                    }
                    for (let i = 0; i < indices.length; i++) {
                        data['selected'][indices[i]] = 1;
                        data['colormapper'][indices[i]] = -42; // number must be lower than lowest cost
                        data['line_width'][indices[i]] = 3;
                        data['line_alpha'][indices[i]] = 1;
                    }
                    source.change.emit();
               //     src2.change.emit();
                """,
        )

        cds.selected.js_on_change("indices", callback)

        return buttons

    def _create_trajectory_debugger(self, main_figure):
        figures = []
        cds = main_figure.select(name="all trajs", tags=["data"])[0]
        main_figure_traj_renderer = main_figure.select(name="all trajs", tags="renderer")[0]
        view = main_figure_traj_renderer.view
        mapper = main_figure_traj_renderer.glyph.line_color
        tools = ["pan", "wheel_zoom", "box_zoom", "tap", "save", "reset"]

        def add_subplot(x_entry: str, y_entry: str, x_label: str, y_label: str):
            fig = figure(
                x_range=(
                    np.array(cds.data[x_entry]).min(),
                    np.array(cds.data[x_entry]).max(),
                ),
                y_range=(
                    np.array(cds.data[y_entry]).min() - 1,
                    np.array(cds.data[y_entry]).max() + 1,
                ),
                x_axis_label=x_label,
                y_axis_label=y_label,
                tools=tools,
                match_aspect=True,
                active_scroll="wheel_zoom",
                margin=[10, 10, 10, 10],
                background_fill_color="white",
                output_backend=main_figure.output_backend,
                sizing_mode="scale_both",
            )
            fig.xgrid.visible = False
            fig.ygrid.visible = False
            glyph = MultiLine(
                xs=x_entry,
                ys=y_entry,
                line_color=mapper,
                line_width="line_width",
                line_alpha="line_alpha",
            )
            fig.add_glyph(cds, glyph, view=view)
            return fig

        # velocity profile
        fig = add_subplot(x_entry="T", y_entry="v", x_label="t in s", y_label="v in m/s")
        fig.line(
            x=[cds.data["T"][0][0], cds.data["T"][0][-1]],
            y=[self.desired_velocity, self.desired_velocity],
            line_color="black",
            line_dash="dashed",
            line_width=1,
        )
        figures.append(fig)

        # orientation
        fig = add_subplot(x_entry="T", y_entry="psi", x_label="t in s", y_label="psi in rad")
        figures.append(fig)

        for fig in figures[2:]:
            fig.x_range = figures[1].x_range

        return figures

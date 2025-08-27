__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import warnings
import shutil
import hydra
import os
import time
import concurrent.futures
from pathlib import Path
from ml_planner.simulation_interfaces.commonroad.commonroad_interface import CommonroadInterface
import csv
import json
import multiprocessing

"""
This script runs batch simulations of multiple CommonRoad scenarios with MP-RBFN Planner.
"""

###############################
# PATH AND DEBUG CONFIGURATION
CWD = Path.cwd()  # This will be the scripts/ directory
PROJECT_ROOT = CWD.parent  # Go up one level to the main project directory

# External Frenetix-style inputs
SCENARIO_ROOT = Path("/home/yuan/Dataset/Frenetix-Motion-Planner/Scenarios_batch")
SCENARIO_LIST_CSV = Path("/home/yuan/Dataset/Frenetix-Motion-Planner/scenario_batch_list.csv")

DATA_PATH = PROJECT_ROOT / "example_scenarios"  # unused when using SCENARIO_LIST_CSV
LOG_PATH = PROJECT_ROOT / "logs"
MODEL_PATH = PROJECT_ROOT / "ml_planner" / "sampling" / "models"

# Batch configuration
DELETE_ALL_FORMER_LOGS = True
MAX_WORKERS = 2  # Number of parallel simulations (GPU memory is limited)
LOGGING_LEVEL_INTERFACE = "info"  # Reduced logging for batch processing
LOGGING_LEVEL_PLANNER = "info"
CPU_ONLY = False  # Set True to force CPU execution and allow many workers

# If CPU_ONLY is enabled, hide CUDA devices for this process and its children
if CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Treat all RuntimeWarnings as errors
warnings.filterwarnings("error", category=RuntimeWarning)
###############################


def get_scenario_list():
    """
    Build list of scenario XML files from a CSV of scenario names using
    Frenetix Scenarios_batch structure:
      <SCENARIO_ROOT>/<name>/Original/<name>.xml

    Returns:
        List[Path]: scenario XML paths that exist
    """
    scenarios: list[Path] = []

    if not SCENARIO_LIST_CSV.exists():
        print(f"❌ Scenario list CSV not found: {SCENARIO_LIST_CSV}")
        return scenarios
    if not SCENARIO_ROOT.exists():
        print(f"❌ Scenario root not found: {SCENARIO_ROOT}")
        return scenarios

    # Read names from CSV (one name per line, ignore comments/empties)
    names: list[str] = []
    with open(SCENARIO_LIST_CSV, 'r') as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith('#'):
                continue
            names.append(name)

    for name in names:
        xml_path = SCENARIO_ROOT / name / "Original" / f"{name}.xml"
        if xml_path.exists():
            scenarios.append(xml_path)
        else:
            print(f"  ⚠️  Missing XML for {name}: {xml_path}")

    if not scenarios:
        print("❌ No valid scenarios resolved from CSV list.")
        return scenarios

    print(f"Found {len(scenarios)} scenarios from CSV:")
    for p in scenarios:
        print(f"  - {p.name}")
    return scenarios


def create_config(scenario_path, log_dir=None):
    """
    Creates a configuration for the simulation.

    Args:
        scenario_path: Path to the scenario file
        log_dir: Optional per-scenario log directory to override default LOG_PATH

    Returns:
        The configuration object for the simulation.
    """
    target_log_path = log_dir if log_dir is not None else LOG_PATH

    # config overrides
    overrides = [
        # general overrides
        f"log_path={target_log_path}",
        # simulation overrides
        f"interface_logging_level={LOGGING_LEVEL_INTERFACE}",
        f"scenario_path={scenario_path}",
        # planner overrides
        f"planner_config.logging_level={LOGGING_LEVEL_PLANNER}",
        f"planner_config.sampling_model_path={MODEL_PATH}",
    ]

    # Compose the configuration
    config_dir = str(Path.cwd().parent / "ml_planner" / "simulation_interfaces" / "commonroad" / "configurations")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        config = hydra.compose(config_name="simulation", overrides=overrides)
    return config


def _ensure_logs_and_csv():
    """Ensure logs directory exists and create CSV header if needed."""
    os.makedirs(LOG_PATH, exist_ok=True)
    overview_path = LOG_PATH / "score_overview.csv"
    if not overview_path.exists():
        with open(overview_path, 'w', newline='') as f:
            f.write("scenario;agent;timestep;status;message;result;collision_type;colliding_object_id\n")
    return overview_path


def _write_scenario_result(scenario_name, status, message, timesteps, result_cost, execution_time, planner_info):
    """Write a single scenario result to the CSV file (Frenetix 8-column format)."""
    overview_path = _ensure_logs_and_csv()
    
    with open(overview_path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        # Frenetix format: scenario;agent;timestep;status;message;result;collision_type;colliding_object_id
        writer.writerow([
            scenario_name,
            "mprbfn",
            timesteps if timesteps is not None else "",
            status,
            message,
            result_cost,
            "",
            "",
        ])


def _create_scenario_folder(scenario_name):
    """Create organized folder structure for a scenario's outputs."""
    scenario_folder = LOG_PATH / scenario_name.replace('.xml', '')
    plots_folder = scenario_folder / "plots"
    
    os.makedirs(scenario_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    
    return scenario_folder, plots_folder


def _save_planner_info(interface, scenario_folder):
    """Save detailed planner information to a JSON file."""
    try:
        planner_info = {
            "scenario_name": getattr(interface, 'scenario_name', 'unknown'),
            "max_timesteps": getattr(interface, 'max_time_steps_scenario', 'unknown'),
            "final_timestep": getattr(interface, 'timestep', 'unknown'),
            "goal_reached": getattr(interface, 'goal_reached', False),
            "device": str(getattr(interface, 'device', 'unknown')),
            "planner_type": "MP-RBFN",
            "sampling_model": str(getattr(interface.trajectory_planner, 'sampling_model', 'unknown')) if hasattr(interface, 'trajectory_planner') else 'unknown',
            "optimal_cost": float(interface.trajectory_planner.optimal_costs_sum.detach().cpu().numpy()) if hasattr(interface, 'trajectory_planner') and hasattr(interface.trajectory_planner, 'optimal_costs_sum') else 'unknown',
            "trajectory_length": len(interface.optimal_trajectory.states) if hasattr(interface, 'optimal_trajectory') else 'unknown'
        }
        
        with open(scenario_folder / "planner_info.json", 'w') as f:
            json.dump(planner_info, f, indent=2)
            
        return json.dumps(planner_info)
    except Exception as e:
        return f"Error collecting planner info: {str(e)}"


def _patch_sim_duration(interface):
    """Patch CommonroadInterface.run() to use scenario goal time instead of 2 steps."""
    original_run = interface.run

    def _extended_run():
        cr_state_global = interface.cr_obstacle_list.pop()
        try:
            # Use CommonRoad goal time end (already in timesteps)
            max_steps = int(interface.planning_problem.goal.state_list[0].time_step.end)
            interface.max_time_steps_scenario = max_steps
            interface.msg_logger.info(f"Scenario goal time detected: {max_steps} timesteps")
        except Exception:
            interface.max_time_steps_scenario = 50
            interface.msg_logger.warning(
                f"Could not detect scenario goal steps. Falling back to {interface.max_time_steps_scenario}."
            )

        while interface.timestep < interface.max_time_steps_scenario:
            interface.msg_logger.debug(f"current timestep {interface.timestep}")
            interface.goal_reached = interface.planning_problem.goal.is_reached(cr_state_global.initial_state)
            if interface.goal_reached:
                interface.msg_logger.info("Goal reached")
                break

            interface.plan_step(cr_state_global)
            cr_state_global = interface.convert_trajectory_to_commonroad_object(interface.optimal_trajectory, interface.timestep)
            interface.cr_obstacle_list.append(cr_state_global)
            interface.visualize_timestep(interface.timestep)

            # prepare next iteration
            from ml_planner.general_utils.data_types import StateTensor
            next_state = StateTensor(
                states=interface.optimal_trajectory.states[1],
                covs=interface.optimal_trajectory.covs[1],
                device=interface.device,
            )
            interface.timestep += 1
            interface.planner_state_list.append(next_state)

        interface.msg_logger.info("Simulation finished")

    interface.run = _extended_run


def _move_plots_to_scenario_folder(scenario_folder, plots_folder):
    """Move all plot files to the scenario-specific plots folder."""
    try:
        # Look for all plot files in the main logs directory
        for plot_file in LOG_PATH.glob("*.png"):
            # Move all PNG files (not just final_trajectory)
            shutil.move(str(plot_file), str(plots_folder / plot_file.name))
            print(f"  📊 Moved plot: {plot_file.name} -> {plots_folder}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not move plot files: {e}")


def _move_logs_to_scenario_folder(scenario_folder):
    """Move log files to the scenario-specific folder."""
    try:
        # Look for log files in the main logs directory
        for log_file in LOG_PATH.glob("*.log"):
            if log_file.name.startswith("Interface_Logger") or log_file.name.startswith("ML_Planner"):
                shutil.move(str(log_file), str(scenario_folder / log_file.name))
                print(f"  📝 Moved log: {log_file.name} -> {scenario_folder}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not move log files: {e}")


def run_single_simulation(scenario_path):
    """
    Run a single simulation for a given scenario.

    Returns a tuple with summary details for CSV writing.
    (scenario_name, status, message, execution_time, timesteps, result_cost)
    """
    scenario_name = scenario_path.name
    start_time = time.time()

    try:
        print(f"\n🚀 Starting simulation for: {scenario_name}")
        
        # Create scenario-specific folder structure
        scenario_folder, plots_folder = _create_scenario_folder(scenario_name)
        
        # Create config with per-scenario log path so interface writes directly there
        config = create_config(scenario_path, log_dir=scenario_folder)
        interface = CommonroadInterface(**config)

        # Respect scenario goal time
        _patch_sim_duration(interface)

        interface.run()

        # Save planner information
        planner_info = _save_planner_info(interface, scenario_folder)

        # Optional visuals (best-effort only) - save to scenario folder
        try:
            interface.plot_final_trajectory()
            print(f"  📊 Generated final trajectory plot")
        except Exception as e:
            print(f"  ⚠️  Plot warning for {scenario_name}: {e}")
            
        try:
            interface.create_gif()
            print(f"  🎬 Generated GIF animation")
        except Exception as e:
            print(f"  ⚠️  GIF warning for {scenario_name}: {e}")

        # Move ALL generated files to scenario folder
        _move_plots_to_scenario_folder(scenario_folder, plots_folder)
        _move_logs_to_scenario_folder(scenario_folder)
        
        # Move GIF files directly to scenario folder
        try:
            for gif_file in LOG_PATH.glob("*.gif"):
                shutil.move(str(gif_file), str(scenario_folder / gif_file.name))
                print(f"  🎬 Moved GIF: {gif_file.name} -> {scenario_folder}")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not move GIF files: {e}")

        execution_time = time.time() - start_time
        timesteps = getattr(interface, 'timestep', None)
        
        # Try to read optimal cost
        try:
            result_cost = float(interface.trajectory_planner.optimal_costs_sum.detach().cpu().numpy())
        except Exception:
            result_cost = ''

        print(f"  ✅ Successfully completed {scenario_name} in {execution_time:.2f}s")
        
        # Write result immediately to CSV
        _write_scenario_result(scenario_name, "success", "", timesteps, result_cost, execution_time, planner_info)
        print(f"  📝 Updated CSV with results for {scenario_name}")
        
        return (scenario_name, "success", "", execution_time, timesteps, result_cost)

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        print(f"  ❌ Failed to run {scenario_name} after {execution_time:.2f}s: {error_msg}")
        
        # Create scenario folder even for failed runs
        scenario_folder, _ = _create_scenario_folder(scenario_name)
        
        # Save error information
        error_info = {
            "error": error_msg,
            "execution_time": execution_time,
            "status": "failed"
        }
        with open(scenario_folder / "error_info.json", 'w') as f:
            json.dump(error_info, f, indent=2)
        
        # Write failure immediately to CSV
        _write_scenario_result(scenario_name, "fail", error_msg, None, '', execution_time, f"Error: {error_msg}")
        print(f"  📝 Updated CSV with failure for {scenario_name}")
        
        return (scenario_name, "fail", error_msg, execution_time, None, '')


def run_batch_simulations(scenario_files, max_workers=2):
    """
    Run simulations for multiple scenarios in parallel.
    
    Args:
        scenario_files: List of scenario file paths
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of results for each scenario
    """
    print(f"\n🎯 Starting batch simulation with {len(scenario_files)} scenarios")
    print(f"🔧 Using {max_workers} parallel workers")
    
    results = []

    # Use spawn start method for better stability with plotting libs
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scenarios
        futures = [executor.submit(run_single_simulation, scenario) for scenario in scenario_files]
        
        # Process completed simulations with timeout to avoid hanging indefinitely
        try:
            for future in concurrent.futures.as_completed(futures, timeout=None):
                try:
                    result = future.result(timeout=1_000)
                except concurrent.futures.TimeoutError:
                    # Skip this one for now; it will reappear in as_completed
                    continue
                except Exception as e:
                    result = ("unknown", "fail", str(e), 0, None, "")
                results.append(result)
        except KeyboardInterrupt:
            print("\n❌ Interrupted by user. Attempting graceful shutdown...")
            for f in futures:
                f.cancel()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    
    return results


def print_batch_summary(results):
    """Print a summary of batch simulation results."""
    print("\n" + "="*80)
    print("📊 BATCH SIMULATION SUMMARY")
    print("="*80)

    successful = [r for r in results if r[1] == "success"]
    failed = [r for r in results if r[1] != "success"]

    print(f"Total scenarios: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")

    if successful:
        total_time = sum(r[3] for r in successful)
        avg_time = total_time / len(successful)
        print(f"\n⏱️  Total execution time: {total_time:.2f}s")
        print(f"⏱️  Average time per scenario: {avg_time:.2f}s")
        print(f"\n✅ Successfully completed scenarios:")
        for name, _, __, time_taken, ___, ____ in successful:
            print(f"  - {name} ({time_taken:.2f}s)")

    if failed:
        print(f"\n❌ Failed scenarios:")
        for name, _, error, __, ___, ____ in failed:
            print(f"  - {name}: {error}")

    print(f"\n📁 Results saved in: {LOG_PATH}")
    print("   Each scenario has its own folder with plots, GIFs, logs, and planner info")


def main():
    """
    Main function to run batch simulations.
    """
    print("🚗 MP-RBFN Batch Simulation Runner")
    print("="*50)
    
    # Clean up logs if requested
    if DELETE_ALL_FORMER_LOGS:
        print("🧹 Cleaning up previous logs...")
        shutil.rmtree(LOG_PATH, ignore_errors=True)
    
    # Get list of available scenarios
    scenario_files = get_scenario_list()
    
    if not scenario_files:
        print("❌ No scenario files found in example_scenarios folder!")
        return
    
    # Ask user for confirmation
    print(f"\n🤔 Found {len(scenario_files)} scenarios. Proceed with batch simulation?")
    print("Press Enter to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Batch simulation cancelled by user.")
        return
    
    # Run batch simulations
    results = run_batch_simulations(scenario_files, MAX_WORKERS)
    
    # Print summary
    print_batch_summary(results)
    
    print("\n🎉 Batch simulation completed!")


if __name__ == "__main__":
    main()

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

"""
This script runs batch simulations of multiple CommonRoad scenarios with MP-RBFN Planner.
Simplified version without user prompts.
"""

###############################
# PATH AND DEBUG CONFIGURATION
CWD = Path.cwd()  # This will be the scripts/ directory
PROJECT_ROOT = CWD.parent  # Go up one level to the main project directory

DATA_PATH = PROJECT_ROOT / "example_scenarios"
LOG_PATH = PROJECT_ROOT / "logs"
MODEL_PATH = PROJECT_ROOT / "ml_planner" / "sampling" / "models"

# Batch configuration - MODIFY THESE SETTINGS AS NEEDED
DELETE_ALL_FORMER_LOGS = False
MAX_WORKERS = 2  # Number of parallel simulations (adjust based on your GPU memory)
LOGGING_LEVEL_INTERFACE = "info"  # Reduced logging for batch processing
LOGGING_LEVEL_PLANNER = "info"
SKIP_PLOTTING = False  # Set to True to skip trajectory plots and GIFs for faster processing

# Note: Simulation timesteps are now automatically read from the scenario XML goal state
# No need to manually configure this - it will use the scenario's intended duration
###############################


def get_scenario_list():
    """
    Get list of all available scenarios in the example_scenarios folder.
    
    Returns:
        List of scenario file paths
    """
    scenario_files = []
    for file in DATA_PATH.glob("*.xml"):
        if file.is_file():
            scenario_files.append(file)
    
    print(f"Found {len(scenario_files)} scenarios:")
    for scenario in scenario_files:
        print(f"  - {scenario.name}")
    
    return scenario_files


def create_config(scenario_path):
    """
    Creates a configuration for the simulation.

    Args:
        scenario_path: Path to the scenario file

    Returns:
        The configuration object for the simulation.
    """
    # config overrides
    overrides = [
        # general overrides
        f"log_path={LOG_PATH}",
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


def modify_simulation_duration(interface):
    """
    Modify the simulation interface to run for the scenario's intended duration.
    This patches the hardcoded 2-timestep limit by reading the goal time from XML.
    
    Args:
        interface: The CommonroadInterface instance
    """
    # Monkey patch the run method to use scenario-defined timesteps
    original_run = interface.run
    
    def extended_run():
        """Extended run method that respects scenario goal time"""
        cr_state_global = interface.cr_obstacle_list.pop()
        
        # Read goal time from scenario XML instead of hardcoded 2 timesteps
        try:
            # Get the goal time from the planning problem using the correct CommonRoad structure
            goal_time_step = interface.planning_problem.goal.state_list[0].time_step.end
            # Convert to timesteps (assuming 0.1s per timestep as per CommonRoad standard)
            interface.max_time_steps_scenario = int(goal_time_step)
            interface.msg_logger.info(f"Scenario goal time: {goal_time_step} timesteps")
        except Exception as e:
            # Fallback to a reasonable default if goal time can't be read
            interface.max_time_steps_scenario = 50  # Default to 50 timesteps (5 seconds)
            interface.msg_logger.warning(f"Could not read goal time from scenario, using default: {interface.max_time_steps_scenario} timesteps")
        
        while interface.timestep < interface.max_time_steps_scenario:
            interface.msg_logger.debug(f"current timestep {interface.timestep}")
            # check if goal reached
            interface.goal_reached = interface.planning_problem.goal.is_reached(cr_state_global.initial_state)
            if interface.goal_reached:
                # simulation finished if goal is reached
                interface.msg_logger.info("Goal reached")
                break

            interface.plan_step(cr_state_global)

            # add current trajectory to list
            cr_state_global = interface.convert_trajectory_to_commonroad_object(interface.optimal_trajectory, interface.timestep)
            interface.cr_obstacle_list.append(cr_state_global)

            # visualize current timestep
            interface.visualize_timestep(interface.timestep)

            # prepare next iteration - use the correct StateTensor class
            from ml_planner.general_utils.data_types import StateTensor
            next_state = StateTensor(
                states=interface.optimal_trajectory.states[1],
                covs=interface.optimal_trajectory.covs[1],
                device=interface.device,
            )

            interface.timestep += 1
            interface.planner_state_list.append(next_state)

        interface.msg_logger.info("Simulation finished")
    
    # Replace the run method
    interface.run = extended_run


def run_single_simulation(scenario_path):
    """
    Run a single simulation for a given scenario.
    
    Args:
        scenario_path: Path to the scenario file
        
    Returns:
        Tuple of (scenario_name, success_status, error_message, execution_time)
    """
    scenario_name = scenario_path.name
    start_time = time.time()
    
    try:
        print(f"\n🚀 Starting simulation for: {scenario_name}")
        
        # Create configuration for this scenario
        config = create_config(scenario_path)
        
        # Create simulation interface
        interface = CommonroadInterface(**config)
        
        # Modify simulation duration
        modify_simulation_duration(interface)
        
        # Run simulation
        interface.run()
        
        # Plot final trajectory (optional - can be disabled for faster batch processing)
        if not SKIP_PLOTTING:
            try:
                interface.plot_final_trajectory()
            except Exception as e:
                print(f"  ⚠️  Warning: Could not create final trajectory plot for {scenario_name}: {e}")
        
        # Create GIF (optional - can be disabled for faster batch processing)
        if not SKIP_PLOTTING:
            try:
                interface.create_gif()
            except Exception as e:
                print(f"  ⚠️  Warning: Could not create GIF for {scenario_name}: {e}")
        
        execution_time = time.time() - start_time
        print(f"  ✅ Successfully completed {scenario_name} in {execution_time:.2f}s")
        
        return (scenario_name, True, None, execution_time)
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        print(f"  ❌ Failed to run {scenario_name} after {execution_time:.2f}s: {error_msg}")
        
        return (scenario_name, False, error_msg, execution_time)


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
    
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scenarios
        future_to_scenario = {
            executor.submit(run_single_simulation, scenario): scenario 
            for scenario in scenario_files
        }
        
        # Process completed simulations
        for future in concurrent.futures.as_completed(future_to_scenario):
            scenario = future_to_scenario[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                scenario_name = scenario.name
                print(f"  ❌ Exception occurred for {scenario_name}: {e}")
                results.append((scenario_name, False, str(e), 0))
    
    return results


def print_batch_summary(results):
    """
    Print a summary of batch simulation results.
    
    Args:
        results: List of simulation results
    """
    print("\n" + "="*80)
    print("📊 BATCH SIMULATION SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"Total scenarios: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        total_time = sum(r[3] for r in successful)
        avg_time = total_time / len(successful)
        print(f"\n⏱️  Total execution time: {total_time:.2f}s")
        print(f"⏱️  Average time per scenario: {avg_time:.2f}s")
        
        print(f"\n✅ Successfully completed scenarios:")
        for name, success, error, time_taken in successful:
            print(f"  - {name} ({time_taken:.2f}s)")
    
    if failed:
        print(f"\n❌ Failed scenarios:")
        for name, success, error, time_taken in failed:
            print(f"  - {name}: {error}")
    
    print("="*80)


def main():
    """
    Main function to run batch simulations.
    """
    print("🚗 MP-RBFN Batch Simulation Runner (Simple)")
    print("="*50)
    
    # Display configuration
    print(f"⚙️  Configuration:")
    print(f"  - Max Workers: {MAX_WORKERS}")
    print(f"  - Simulation Timesteps: Auto-detected from scenario XML")
    print(f"  - Skip Plotting: {SKIP_PLOTTING}")
    print()
    
    # Clean up logs if requested
    if DELETE_ALL_FORMER_LOGS:
        print("🧹 Cleaning up previous logs...")
        shutil.rmtree(LOG_PATH, ignore_errors=True)
    
    # Get list of available scenarios
    scenario_files = get_scenario_list()
    
    if not scenario_files:
        print("❌ No scenario files found in example_scenarios folder!")
        return
    
    # Run batch simulations
    results = run_batch_simulations(scenario_files, MAX_WORKERS)
    
    # Print summary
    print_batch_summary(results)
    
    print("\n🎉 Batch simulation completed!")
    print(f"📁 Results saved in: {LOG_PATH}")
    print(f"🎬 GIFs will now show the full scenario duration instead of just 2 timesteps!")
    print(f"⏱️  Each scenario will run for its intended goal time (e.g., 147 timesteps = 14.7s)")


if __name__ == "__main__":
    main()

[![Linux](https://img.shields.io/badge/os-linux-blue.svg)](https://www.linux.org/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) 
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)

# MP-RBFN - Radial-basis function based Motion Primitives

> **⚠️ Important Note**: This repository is a **modified fork** of the original [MP-RBFN repository](https://github.com/TUM-AVS/MP-RBFN) with significant enhancements for batch simulation and Frenetix integration.

## Repository Status

This is a **forked and modified version** that includes:
- **Enhanced batch simulation system** with multi-process execution
- **Frenetix-style scenario integration** for seamless workflow
- **Improved visualization** with 60m x 60m ego-centered plots
- **Real-time CSV output** following Frenetix format standards
- **Comprehensive error handling** and troubleshooting
- **Performance optimizations** for CPU-based parallel execution

## Updating from Original Repository

### If you want to pull latest changes from the original repository:

```bash
# Add the original repository as upstream
git remote add upstream https://github.com/TUM-AVS/MP-RBFN.git

# Fetch latest changes
git fetch upstream

# Merge upstream changes (resolve conflicts if any)
git merge upstream/main

# Push to your fork
git push origin main
```

### ⚠️ **Warning**: 
- **Backup your modifications** before updating
- **Review all conflicts** carefully - our batch simulation features may conflict with upstream changes
- **Test thoroughly** after merging to ensure batch functionality still works
- **Consider creating a feature branch** for major upstream updates

### Recommended Workflow:
1. **Create a backup branch** of your current modifications
2. **Update from upstream** on a separate branch
3. **Merge selectively** to preserve batch simulation features
4. **Test thoroughly** before updating your main branch

## Original Repository

The original MP-RBFN repository can be found at:
- **GitHub**: [TUM-AVS/MP-RBFN](https://github.com/TUM-AVS/MP-RBFN)
- **Original Authors**: Marc Kaufeld, Mattia Piccinini, Johannes Betz
- **Institution**: Technical University of Munich (TUM)
---

This repository contains the implementation of MP-RBFN (Motion Primitives based on Radial Basis Function Networks) for motion planning in autonomous driving scenarios.

## Installation

### Option 1: Using conda (Recommended)

```bash
# Create and activate conda environment
conda create -n mprbfn python=3.11
conda activate mprbfn

# Install the package in editable mode
pip install -e .
```

### Option 2: Using venv

```bash
# Create and activate virtual environment
python3.11 -m venv mprbfn_env
source mprbfn_env/bin/activate  # On Windows: mprbfn_env\Scripts\activate

# Install the package in editable mode
pip install -e .
```

## Quick Start

### Single Scenario Simulation

Run a single CommonRoad scenario:

```bash
cd scripts
python run_cr_simulation.py
```

### Batch Simulation

Run multiple scenarios in parallel with organized output:

```bash
cd scripts
python run_cr_simulation_batch.py
```

## Batch Simulation System

### Overview

The batch simulation system allows you to process multiple CommonRoad scenarios in parallel, with organized output management and real-time progress tracking.

### Features

- **Multi-process execution**: Run up to 12+ scenarios simultaneously (CPU-based)
- **Organized outputs**: Each scenario gets its own folder with plots, GIFs, logs, and metadata
- **Real-time CSV updates**: Results are written after each scenario completion
- **Frenetix-style integration**: Compatible with existing scenario folder structures
- **Automatic scenario detection**: Reads scenario list from CSV file
- **Comprehensive logging**: Detailed planner information and error handling

### Scenario Structure

The batch system expects scenarios organized in Frenetix-style structure:

```
/home/yuan/Dataset/Frenetix-Motion-Planner/Scenarios_batch/
├── CHN_Beijing-7_7_T-1/
│   └── Original/
│       └── CHN_Beijing-7_7_T-1.xml
├── CHN_Qingdao-11_13_T-1/
│   └── Original/
│       └── CHN_Qingdao-11_13_T-1.xml
├── DEU_Arnstadt-36_1_T-1/
│   └── Original/
│       └── DEU_Arnstadt-36_1_T-1.xml
└── ... (more scenario folders)
```

### Configuration

#### 1. Edit the CSV file

Modify `/home/yuan/Dataset/Frenetix-Motion-Planner/scenario_batch_list.csv` to include only the scenarios you want to process:

```csv
CHN_Beijing-7_7_T-1
CHN_Qingdao-11_13_T-1
DEU_Arnstadt-36_1_T-1
# Add or remove scenarios as needed
```

#### 2. Run batch evaluation

```bash
cd /home/yuan/Dataset/RBFN-Motion-Primitives/scripts
python run_cr_simulation_batch.py
```

#### 3. Configuration options

In `run_cr_simulation_batch.py`, you can modify:

```python
# Batch configuration
DELETE_ALL_FORMER_LOGS = True
MAX_WORKERS = 12  # Number of parallel simulations
LOGGING_LEVEL_INTERFACE = "info"
LOGGING_LEVEL_PLANNER = "info"
CPU_ONLY = True  # Force CPU execution for higher parallelism
```

### Output Structure

After running batch simulations, results are organized as follows:

```
logs/
├── score_overview.csv                    # Summary of all scenarios
├── CHN_Beijing-7_7_T-1/
│   ├── plots/                           # All generated plots
│   │   ├── CHN_Beijing-7_7_T-1_0.png
│   │   ├── CHN_Beijing-7_7_T-1_1.png
│   │   └── final_trajectory.png
│   ├── CHN_Beijing-7_7_T-1.gif         # Animation GIF
│   ├── Interface_Logger.log             # Interface logs
│   ├── ML_Planner.log                   # Planner logs
│   └── planner_info.json                # Detailed planner metrics
├── CHN_Qingdao-11_13_T-1/
│   ├── plots/
│   ├── CHN_Qingdao-11_13_T-1.gif
│   ├── logs/
│   └── planner_info.json
└── ...
```

### CSV Output Format

The `score_overview.csv` follows Frenetix format:

```csv
scenario;agent;timestep;status;message;result;collision_type;colliding_object_id
CHN_Beijing-7_7_T-1;mprbfn;147;success;;0.892929;;
CHN_Qingdao-11_13_T-1;mprbfn;80;success;;1.159697;;
DEU_Arnstadt-36_1_T-1;mprbfn;;fail;CUDA out of memory;;;
```

### Performance Optimization

#### CPU vs GPU Mode

- **GPU Mode** (`CPU_ONLY = False`): Faster per scenario, limited to 1-2 workers
- **CPU Mode** (`CPU_ONLY = True`): Slower per scenario, but can use 12+ workers

#### Memory Management

- **GPU Mode**: Set `MAX_WORKERS = 1` or `2` to avoid CUDA OOM
- **CPU Mode**: Set `MAX_WORKERS = os.cpu_count()` for maximum parallelism

## Visualization Features

### Plot Configuration

- **60m x 60m window**: Each timestep plot is centered on the ego vehicle with a 60m x 60m view
- **SVG/PNG output**: Configurable via `visualization_config.make_plot` and `make_gif`
- **Frenetix-style focus**: Consistent with Frenetix visualization standards

### Disable Unwanted Outputs

To reduce file generation and improve performance:

```python
# In create_config overrides
overrides = [
    # ... other overrides ...
    "visualization_config.make_plot=false",  # Disable SVG plots
    "visualization_config.make_gif=true",   # Keep GIFs for animation
]
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Problem**: `CUDA out of memory` errors during batch execution

**Solution**: 
- Set `CPU_ONLY = True` in the batch script
- Reduce `MAX_WORKERS` to 1-2 if using GPU
- Ensure no other GPU processes are running

#### 2. Script Hangs After Simulation

**Problem**: Script doesn't exit automatically after completing scenarios

**Solution**: 
- Ensure matplotlib figures are properly closed (already implemented)
- Use `spawn` start method for multiprocessing
- Check for stuck worker processes

#### 3. Plots Not Saved in Scenario Folders

**Problem**: Plots are saved in main logs directory instead of scenario-specific folders

**Solution**: 
- Verify `log_dir` parameter is passed correctly to `create_config`
- Check file permissions for scenario folders
- Ensure plots are moved after generation

#### 4. CSV Not Updating in Real-time

**Problem**: `score_overview.csv` only updates at the end instead of after each scenario

**Solution**: 
- Verify `_write_scenario_result` is called after each scenario
- Check file write permissions
- Ensure CSV file path is correct

#### 5. GIF Shows Only Part of Scenario

**Problem**: Generated GIFs only show 1-2 timesteps instead of full scenario

**Solution**: 
- Verify `_patch_sim_duration` is correctly reading goal time from XML
- Check that `interface.max_time_steps_scenario` is set correctly
- Ensure simulation runs for full duration

### Debug Mode

Enable detailed logging for troubleshooting:

```python
LOGGING_LEVEL_INTERFACE = "debug"
LOGGING_LEVEL_PLANNER = "debug"
```

### Error Logs

Failed scenarios create detailed error logs:

```
logs/
└── FAILED_SCENARIO_NAME/
    ├── error_info.json          # Error details and execution time
    └── planner_info.json        # Available planner state at failure
```

## Advanced Configuration

### Custom Scenario Paths

To use different scenario locations:

```python
# In run_cr_simulation_batch.py
SCENARIO_ROOT = Path("/path/to/your/Scenarios_batch")
SCENARIO_LIST_CSV = Path("/path/to/your/scenario_list.csv")
```

### Custom Output Paths

Modify output locations:

```python
LOG_PATH = Path("/custom/logs/path")
MODEL_PATH = Path("/custom/model/path")
```

### Environment Variables

Set environment variables for specific configurations:

```bash
# Force CPU-only execution
export CUDA_VISIBLE_DEVICES=-1

# Set custom log levels
export INTERFACE_LOG_LEVEL=debug
export PLANNER_LOG_LEVEL=debug
```

## Performance Tips

1. **Use CPU mode** for high-parallelism batch runs
2. **Disable SVG plots** if only GIFs are needed
3. **Reduce logging levels** for production runs
4. **Monitor memory usage** when increasing worker count
5. **Use SSD storage** for faster I/O during batch processing

## Contributing

When modifying the batch system:

1. Test with a small number of scenarios first
2. Verify CSV output format remains compatible
3. Check that all outputs are properly organized
4. Ensure error handling covers edge cases
5. Update this README with new features

## Support

For issues related to:
- **Batch execution**: Check troubleshooting section above
- **Individual scenarios**: Use single scenario mode for debugging
- **Performance**: Monitor system resources and adjust worker count
- **Visualization**: Verify plot configuration and file permissions

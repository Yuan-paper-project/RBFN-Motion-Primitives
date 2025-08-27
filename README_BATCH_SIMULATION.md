# 🚗 MP-RBFN Batch Simulation Guide

This guide explains how to run multiple CommonRoad scenarios in batch using the MP-RBFN planner.

## 📁 Available Scripts

### 1. `run_cr_simulation_batch.py` - Interactive Batch Runner
- **Features**: User confirmation prompts, detailed progress tracking
- **Best for**: Development, testing, when you want control over execution
- **Usage**: Run and confirm before starting batch processing

### 2. `run_cr_simulation_batch_simple.py` - Simple Batch Runner  
- **Features**: No user prompts, runs immediately, faster execution
- **Best for**: Production runs, automated scripts, CI/CD pipelines
- **Usage**: Run and it starts processing immediately

## 🎯 What It Does

The batch scripts will:
1. **Scan** the `example_scenarios/` folder for all `.xml` files
2. **Run** MP-RBFN simulations on each scenario in parallel
3. **Generate** trajectory plots and GIFs for each scenario
4. **Provide** detailed progress updates and timing information
5. **Create** a comprehensive summary report

## ⚙️ Configuration

Edit the configuration section in your chosen script:

```python
# Batch configuration - MODIFY THESE SETTINGS AS NEEDED
DELETE_ALL_FORMER_LOGS = False      # Clean up previous logs
MAX_WORKERS = 2                     # Parallel simulations (adjust for GPU memory)
LOGGING_LEVEL_INTERFACE = "info"    # Logging verbosity
LOGGING_LEVEL_PLANNER = "info"      # Planner logging level
SKIP_PLOTTING = False               # Skip plots/GIFs for speed
```

## 🚀 Running Batch Simulations

### Option 1: Interactive Batch Runner
```bash
cd scripts
python run_cr_simulation_batch.py
```

### Option 2: Simple Batch Runner
```bash
cd scripts
python run_cr_simulation_batch_simple.py
```

## 📊 Expected Output

```
🚗 MP-RBFN Batch Simulation Runner (Simple)
==================================================
Found 8 scenarios:
  - ZAM_Over-1_1_dynamic_1vehicle_10m-s.xml
  - ZAM_Over-1_1_dynamic_1vehicle_15m-s.xml
  - ZAM_Over-1_1_dynamic_1vehicle_5m-s.xml
  - ZAM_Tjunction-1_23_T-1.xml
  - ZAM_Tjunction-1_24_T-1.xml
  - ZAM_Tjunction-1_27_T-1.xml
  - ZAM_Tjunction-1_36_T-1.xml
  - ZAM_Tjunction-1_42_T-1.xml

🎯 Starting batch simulation with 8 scenarios
🔧 Using 2 parallel workers

🚀 Starting simulation for: ZAM_Tjunction-1_27_T-1.xml
  ✅ Successfully completed ZAM_Tjunction-1_27_T-1.xml in 12.34s

🚀 Starting simulation for: ZAM_Over-1_1_dynamic_1vehicle_10m-s.xml
  ✅ Successfully completed ZAM_Over-1_1_dynamic_1vehicle_10m-s.xml in 8.76s

...

=====================================
📊 BATCH SIMULATION SUMMARY
=====================================
Total scenarios: 8
✅ Successful: 8
❌ Failed: 0

⏱️  Total execution time: 89.45s
⏱️  Average time per scenario: 11.18s

✅ Successfully completed scenarios:
  - ZAM_Tjunction-1_27_T-1.xml (12.34s)
  - ZAM_Over-1_1_dynamic_1vehicle_10m-s.xml (8.76s)
  ...

🎉 Batch simulation completed!
```

## 🔧 Performance Tuning

### GPU Memory Considerations
- **Start with `MAX_WORKERS = 1`** if you encounter CUDA out-of-memory errors
- **Increase gradually** to find the optimal parallelization for your GPU
- **Monitor GPU memory usage** during execution

### Speed vs. Quality Trade-offs
- **Set `SKIP_PLOTTING = True`** for faster execution without visualization
- **Reduce logging levels** for less verbose output
- **Use `DELETE_ALL_FORMER_LOGS = True`** to start fresh each time

## 📁 Output Structure

After batch execution, you'll find:

```
logs/
├── plots/
│   ├── ZAM_Tjunction-1_27_T-1_0.svg
│   ├── ZAM_Tjunction-1_27_T-1_1.svg
│   ├── ZAM_Tjunction-1_27_T-1_0.png
│   ├── ZAM_Tjunction-1_27_T-1_1.png
│   └── ...
├── final_trajectory.svg
└── ZAM_Tjunction-1_27_T-1.gif
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `MAX_WORKERS` to 1
   - Close other GPU applications
   - Restart Python environment

2. **Scenario Loading Errors**
   - Check XML file validity
   - Verify file paths are correct
   - Ensure CommonRoad compatibility

3. **Visualization Errors**
   - Set `SKIP_PLOTTING = True` for headless environments
   - Check matplotlib backend compatibility

### Performance Tips

- **Parallel processing**: Use multiple workers for faster execution
- **Logging levels**: Use "info" or "warning" for production runs
- **Memory management**: Monitor GPU memory usage and adjust workers accordingly

## 🔄 Customization

### Adding New Scenarios
Simply place new `.xml` files in the `example_scenarios/` folder - they'll be automatically detected and processed.

### Custom Configuration
Modify the `create_config()` function to add scenario-specific overrides or custom parameters.

### Integration with Other Tools
The batch scripts can be easily integrated with:
- CI/CD pipelines
- Automated testing frameworks
- Performance benchmarking tools
- Research experiment automation

## 📚 Related Files

- `run_cr_simulation.py` - Single scenario runner
- `example_scenarios/` - CommonRoad scenario files
- `ml_planner/simulation_interfaces/commonroad/configurations/` - Configuration files
- `logs/` - Output directory for results

---

**Happy batch simulating! 🚗✨**

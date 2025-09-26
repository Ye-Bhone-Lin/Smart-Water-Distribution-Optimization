
# Smart Water Distribution Optimization

This repository provides a modular framework for optimizing, monitoring, and maintaining smart water distribution systems. It integrates machine learning, network modeling, and predictive maintenance to address real-world water management challenges.

## Repository Structure

- **app.py, main.py**: Entry points for running the application and orchestrating workflows.
- **leak_prediction/**: Leak detection and prediction using machine learning.
	- `leak_model_work.py`, `leak_pred.ipynb`: Model training, evaluation, and analysis.
	- `leak_detection_model.pkl`: Trained leak detection model.
	- `data/`: Datasets for leak detection.
	- `images/`, `workflow/`: Visualizations and process diagrams.
- **water_network_optimization/**: Water network modeling and optimization.
	- `water_network_modelling.py`: Core modeling logic.
	- `create_synthetic_model.py`: Synthetic network generation.
	- `Insein_synthetic_network.inp`: Example EPANET input file.
	- `requirements.txt`: Dependencies for this module.
	- `cache/`: Cached results and intermediate data.
- **water_pump_maintenance/**: Predictive maintenance for water pumps.
	- `pump_model_work.py`, `water_pump_maint.ipynb`: Model development and analysis.
	- `rf_model.pkl`: Trained pump maintenance model.
	- `data/`: Sensor and test data.
	- `workflow/`: Maintenance process diagrams.
- **water_update/**: Update and communication tools for water system status.
	- `water_chat.py`: Chat-based interface for updates.
	- `workflow/`: Update process diagrams.

## Key Features

- **Leak Detection**: Machine learning models for early leak identification using historical and real-time data.
- **Network Optimization**: Simulation and optimization of water distribution networks, including synthetic data generation.
- **Pump Maintenance**: Predictive analytics for pump failure and maintenance scheduling.
- **System Updates**: Tools for communicating system status and updates.

## Getting Started

1. **Install dependencies**:
	 - Use the root `Pipfile` or module-specific `requirements.txt` files.
	 - Example: `pip install -r water_network_optimization/requirements.txt`
2. **Run main application**:
	 - `python app.py` or `python main.py`
3. **Explore modules**:
	 - Notebooks and scripts in each module demonstrate workflows and model usage.

## Data & Models

- Data files are located in each module's `data/` directory.
- Pretrained models are provided as `.pkl` files for immediate use.

## Visualizations

- Diagrams and images in `images/` and `workflow/` folders illustrate data flows and processes.

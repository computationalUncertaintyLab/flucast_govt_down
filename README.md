# Influenza Hospitalization Forecasting Pipeline

## Overview

This project provides an automated pipeline for forecasting influenza hospitalizations using the TEMPO model. The pipeline downloads the latest data, runs historical model fits, generates current season forecasts, and produces visualizations.

## Pipeline Workflow

The complete forecasting pipeline is managed through the `Makefile` and consists of the following stages:

### 1. Environment Setup (`build_env`)
Creates a Python virtual environment (`.forecast/`) and installs all required dependencies from `requirements.txt`.

```bash
make build_env
```

### 2. Data Collection

The pipeline downloads multiple data sources required for forecasting:

#### Download Clinical Data (`download_clinical_data`)
- Downloads public laboratory data using R script
- Formats and processes the lab data
- Outputs: `analysis_data/clinical_and_public_lab_data__formatted.csv`

```bash
make download_clinical_data
```

#### Download ILI Data (`download_ili`)
- Downloads recent ILINet (Influenza-Like Illness Network) data
- Outputs: `analysis_data/ili_data_all_states_2021_present.csv`

```bash
make download_ili
```

#### Download Hospital Reporting Data (`download_hosp_pct_data`)
- Downloads NHSN percent hospital reporting data
- Outputs: `analysis_data/pct_hospital_reporting.csv`

```bash
make download_hosp_pct_data
```

#### Download Weather Data (`download_weather_data`)
- Downloads weather data that may influence flu transmission

```bash
make download_weather_data
```

**Shortcut to run all data downloads:**
```bash
make run_data
```

### 3. Historical Forecasts (`run_historical_forecasts`)

Runs the TEMPO model on historical flu seasons to estimate model parameters and validate performance:
- Processes all past seasons for all states
- Combines results into parameter estimates
- Outputs: `historical_model_run_for_tempo/all_past_param_estimates__tempo4.csv`

```bash
make run_historical_forecasts
```

### 4. Current Season Forecasts (`run_current_season_forecasts`)

Generates forecasts for the current flu season:
- Uses the TEMPO model with historical parameters
- Produces individual state-level forecasts
- Combines all forecasts into a single file
- Outputs: Individual forecasts in `forecasts/` directory and combined timestamped file in `time_stamped_forecasts/`

```bash
make run_current_season_forecasts
```

### 5. Visualization (`visualize_state_level_forecasts`)

Creates visualizations of state-level forecasts for all locations.

```bash
make visualize_state_level_forecasts
```

### Complete Pipeline

To run the entire pipeline from start to finish:

```bash
make forecast
```

This single command executes all stages in the correct order:
1. Build environment
2. Download all data sources
3. Run historical forecasts
4. Generate current season forecasts
5. Create visualizations

## Web Application

### Overview

The project includes an interactive web application built with Streamlit that displays influenza hospitalization forecasts. The app allows users to select multiple locations and view forecasts with prediction intervals.

### Features

- Interactive location selection (US states and national level)
- Median forecast lines with 50% and 80% prediction intervals
- Dynamic visualization that adjusts based on number of selected locations
- Real-time data from AWS S3 storage

### Running the Web App Locally

1. Navigate to the webapp directory:
```bash
cd webapp
```

2. Install webapp-specific requirements:
```bash
pip install -r requirements.txt
```

3. Set up AWS credentials (required for data access):
Create a `.streamlit/secrets.toml` file with your AWS credentials:
```toml
AWS_ACCESS_KEY_ID = "your_access_key"
AWS_SECRET_ACCESS_KEY = "your_secret_key"
```

4. Run the Streamlit app:
```bash
streamlit run main.py
```

5. The app will open automatically in your web browser (typically at `http://localhost:8501`)

### Accessing the Deployed App

If deployed, the web app can be accessed at the URL provided by your Streamlit hosting service.

## Project Structure

- `analysis_data/`: Scripts and data for downloading and formatting source data
- `data/`: Reference data including location information and target data
- `forecasts/`: Individual forecast files for each location
- `time_stamped_forecasts/`: Combined forecast files with timestamps
- `historical_model_run_for_tempo/`: Historical model runs and parameter estimates
- `model/tempo/`: TEMPO model implementation
- `webapp/`: Streamlit web application for visualizing forecasts
- `Makefile`: Automated pipeline workflow
- `requirements.txt`: Python dependencies for forecasting pipeline

## Requirements

- Python 3.x
- R (for downloading clinical data)
- See `requirements.txt` for Python package dependencies
- See `webapp/requirements.txt` for web app dependencies

## Notes

- The pipeline uses the TEMPO model (version 4) for forecasting
- Forecasts are generated for all US states and national level
- Data sources include CDC's ILINet, NHSN hospital data, and public laboratory data
- The virtual environment is created in `.forecast/` directory

## Contact

**Thomas McAndrew**  
Associate Professor  
Department of Biostatistics  
Lehigh University

Email: mcandrew@lehigh.edu  
Lab Website: [Computational Uncertainty Lab](https://compuncertlab.org/)

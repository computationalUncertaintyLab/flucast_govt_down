#mcandrew

#--Set up environment variables
PYTHON ?= python3 -W ignore
R ?= Rscript

#--Set up virtual environment
VENV_DIR := .forecast
VENV_PYTHON := $(VENV_DIR)/bin/python -W ignore

#--Run everything
forecast: build_env download_ili download_clinical_data download_hosp_pct_data download_weather_data run_historical_forecasts run_current_season_forecasts visualize_state_level_forecasts

#--run data
run_data: build_env download_clinical_data download_ili download_hosp_pct_data download_weather_data

#--Build environment
build_env:
	@echo "build forecast environment"
	@$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install -r requirements.txt

#--Download data
download_clinical_data:
	@echo "Downloading Public lab data"
	@$(R) ./analysis_data/download_lab_percentage_data.R
	@$(VENV_PYTHON) ./analysis_data/format_lab_data.py

download_ili:
	@echo "Downloading recent ILINet data"
	@$(VENV_PYTHON) ./analysis_data/build_ili_data.py

download_hosp_pct_data:
	@echo "Downloading NHSNpct hosp data"
	@$(VENV_PYTHON) ./analysis_data/download_percent_reported_hosps.py

download_weather_data:
	@echo "Download weather data"
	@$(VENV_PYTHON) ./analysis_data/download_weather_data.py

#--Build historical forecasts
run_historical_forecasts:
	@echo "Build historical forecasts"
	@$(VENV_PYTHON) ./historical_model_run_for_tempo/run_all_seasons_all_states__tempo4.py
	@$(VENV_PYTHON) ./historical_model_run_for_tempo/combine_all_files.py

run_current_season_forecasts:
	@echo "Build current season forecasts"
	@$(VENV_PYTHON) ./forecast_this_season.py
	@$(VENV_PYTHON) combine_forecasts.py

#--Visuals
visualize_state_level_forecasts:
	@echo "Visualize state level forecasts"
	@$(VENV_PYTHON) ./viz/plot_all_forecasts/plot.py
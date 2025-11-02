#mcandrew

import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import boto3
from epiweeks import Week
from datetime import datetime

# AWS S3 configuration
bucket_name = 'flu-forecast-inc-nodata'
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# Cache data loading from S3
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data_from_s3():
    """Load all datasets from S3 with caching"""
    # Load forecast data
    response = s3.get_object(Bucket=bucket_name, Key='forecasts__2025-11-01.csv')
    forecasts = pd.read_csv(response['Body'])
    
    # Load locations data
    response = s3.get_object(Bucket=bucket_name, Key="locations.csv")
    locations = pd.read_csv(response['Body'])
    
    # Load additional datasets for supporting plots
    response = s3.get_object(Bucket=bucket_name, Key='target-hospital-admissions.csv')
    target_admissions = pd.read_csv(response['Body'])
    
    response = s3.get_object(Bucket=bucket_name, Key='ili_data_all_states_2021_present.csv')
    ili_data = pd.read_csv(response['Body'])
    
    response = s3.get_object(Bucket=bucket_name, Key='weekly_weather_data.csv')
    weather_data = pd.read_csv(response['Body'])
    
    response = s3.get_object(Bucket=bucket_name, Key='pct_hospital_reporting.csv')
    hosp_reporting = pd.read_csv(response['Body'])
    
    return forecasts, locations, target_admissions, ili_data, weather_data, hosp_reporting

# Load all data with caching
forecasts, locations, target_admissions, ili_data, weather_data, hosp_reporting = load_data_from_s3()

# Helper function for location formatting (used in data processing)
def format_location(row):
    loc = str(row['location'])
    # Handle NaN, None, or 'nan' string values
    if loc in ['nan', 'None', ''] or pd.isna(row['location']):
        return None
    if loc == "US" or loc == "us":
        return "US"
    else:
        try:
            return "{:02d}".format(int(float(loc)))
        except (ValueError, TypeError):
            return None

# Helper function to add epidemic week information for seasonal overlay
def add_epiweek_info(df, date_col):
    """
    Add epidemic week information to dataframe for seasonal overlay plotting.
    Seasons run from epi week 40 to week 20 of next year (33-34 weeks total, depending on 52/53 week years).
    Returns df with epiweek, season, and weeks_from_start columns.
    """
    df = df.copy()
    
    # Calculate epidemic week from date
    def get_epiweek_info(date):
        if pd.isna(date):
            return pd.Series({'epiweek': None, 'epiyear': None, 'week_num': None})
        epiweek = Week.fromdate(date)
        return pd.Series({
            'epiweek': epiweek.week,
            'epiyear': epiweek.year,
            'week_num': epiweek.week
        })
    
    # Convert date column to datetime (use single brackets to get Series)
    df[date_col] = pd.to_datetime(df[date_col])
    epiweek_info = df[date_col].apply(get_epiweek_info)
    df['epiweek'] = epiweek_info['epiweek']
    df['epiyear'] = epiweek_info['epiyear']
    
    # Filter to only include epi weeks 40-53 and 1-20 (the flu season)
    # For 2025, also include weeks 35-39 to capture early 2025/2026 season data
    df = df[
        (df['epiweek'] >= 40) | 
        (df['epiweek'] <= 20) |
        ((df['epiyear'] == 2025) & (df['epiweek'] >= 35) & (df['epiweek'] <= 39))
    ].copy()
    
    # Determine season (year season starts) and handle 52/53 week years
    def calculate_season_and_weeks(row):
        epiweek = row['epiweek']
        epiyear = row['epiyear']
        
        # Special handling for 2025 weeks 35-39 (early 2025/2026 season)
        if epiyear == 2025 and 35 <= epiweek <= 39:
            season = "2025/2026"
            weeks_from_start = epiweek - 40  # Will be negative, but that's ok for display
        elif epiweek >= 40:
            # This is the start of the season
            season_start_year = epiyear
            season = f"{epiyear}/{epiyear+1}"
            weeks_from_start = epiweek - 40
        else:
            # This is weeks 1-20 of the next calendar year
            season_start_year = epiyear - 1
            season = f"{epiyear-1}/{epiyear}"
            
            # Check if the season start year had 53 weeks
            try:
                last_week_of_season_start = Week.fromdate(datetime(season_start_year, 12, 28))
                has_53_weeks = last_week_of_season_start.week == 53
            except:
                has_53_weeks = False
            
            # Calculate weeks from start accounting for 52 vs 53 week years
            if has_53_weeks:
                weeks_from_start = epiweek + 14  # weeks 40-53 = 14 weeks, so week 1 starts at position 14
            else:
                weeks_from_start = epiweek + 13  # weeks 40-52 = 13 weeks, so week 1 starts at position 13
        
        return pd.Series({
            'season': season,
            'weeks_from_start': weeks_from_start
        })
    
    season_info = df.apply(calculate_season_and_weeks, axis=1)
    df['season'] = season_info['season']
    df['weeks_from_start'] = season_info['weeks_from_start']
    
    return df

# Cache the data processing pipeline
@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_all_data(forecasts_raw, locations_raw, target_admissions_raw, ili_data_raw, weather_data_raw, hosp_reporting_raw):
    """Process all datasets with caching to avoid recomputation"""
    
    # Make copies to avoid modifying cached data
    forecasts = forecasts_raw.copy()
    locations = locations_raw.copy()
    target_admissions = target_admissions_raw.copy()
    ili_data = ili_data_raw.copy()
    weather_data = weather_data_raw.copy()
    hosp_reporting = hosp_reporting_raw.copy()
    
    # Filter for reference_date 2025-11-01
    forecasts = forecasts[forecasts['reference_date'] == '2025-11-01'].copy()
    
    # Convert location to string for merging
    forecasts['location'] = forecasts['location'].astype(str)
    locations['location'] = locations['location'].astype(str)
    forecasts['location'] = forecasts.apply(lambda x: format_location(x), axis=1)
    locations['location'] = locations.apply(lambda x: format_location(x), axis=1)
    
    # Merge to get location names
    forecasts = forecasts.merge(locations[['location', 'location_name']], on='location', how='left')
    
    # Prepare additional datasets for filtering
    
    # Format target_admissions locations (already has location_name in CSV)
    target_admissions['location'] = target_admissions['location'].astype(str)
    target_admissions['location'] = target_admissions.apply(lambda x: format_location(x), axis=1)
    # Drop rows with invalid locations
    target_admissions = target_admissions.dropna(subset=['location'])
    # Add epidemic week information for seasonal overlay
    target_admissions = add_epiweek_info(target_admissions, 'date')
    
    # Format ILI data locations (has state_name, rename to location_name)
    ili_data['location'] = ili_data['state_fips'].astype(str)
    # Handle US national level (state_fips = -1)
    ili_data.loc[ili_data['state_fips'] == -1, 'location'] = 'US'
    # Format state FIPS codes
    ili_data.loc[ili_data['state_fips'] != -1, 'location'] = ili_data.loc[ili_data['state_fips'] != -1].apply(lambda x: format_location(x), axis=1)
    # Rename state_name to location_name for consistency
    ili_data['location_name'] = ili_data['state_name']
    # ILI data already has epiweek info, just need to process it
    ili_data['epiweek'] = ili_data['week'].astype(int)
    ili_data['epiyear'] = ili_data['year'].astype(int)
    # Filter to only include epi weeks 40-53 and 1-20
    ili_data = ili_data[(ili_data['epiweek'] >= 40) | (ili_data['epiweek'] <= 20)].copy()
    
    # Calculate season and weeks_from_start accounting for 52/53 week years
    def calculate_ili_season_and_weeks(row):
        epiweek = row['epiweek']
        epiyear = row['epiyear']
        
        if epiweek >= 40:
            season = f"{epiyear}/{epiyear+1}"
            weeks_from_start = epiweek - 40
        else:
            season_start_year = epiyear - 1
            season = f"{epiyear-1}/{epiyear}"
            
            # Check if the season start year had 53 weeks
            try:
                last_week_of_season_start = Week.fromdate(datetime(season_start_year, 12, 28))
                has_53_weeks = last_week_of_season_start.week == 53
            except:
                has_53_weeks = False
            
            if has_53_weeks:
                weeks_from_start = epiweek + 14
            else:
                weeks_from_start = epiweek + 13
        
        return pd.Series({
            'season': season,
            'weeks_from_start': weeks_from_start
        })
    
    ili_season_info = ili_data.apply(calculate_ili_season_and_weeks, axis=1)
    ili_data['season'] = ili_season_info['season']
    ili_data['weeks_from_start'] = ili_season_info['weeks_from_start']
    
    # Format weather data locations (already has location_name in CSV)
    weather_data['location'] = weather_data['location'].astype(str)
    weather_data['location'] = weather_data.apply(lambda x: format_location(x), axis=1)
    weather_data = weather_data.dropna(subset=['location'])
    # Add epidemic week information for seasonal overlay
    weather_data = add_epiweek_info(weather_data, 'enddate')
    
    # Format hospital reporting data locations
    hosp_reporting['location'] = hosp_reporting['location'].astype(str)
    hosp_reporting['location'] = hosp_reporting.apply(lambda x: format_location(x), axis=1)
    hosp_reporting = hosp_reporting.dropna(subset=['location'])
    hosp_reporting = hosp_reporting.merge(locations[['location', 'location_name']], on='location', how='left')
    # Add epidemic week information for seasonal overlay
    hosp_reporting = add_epiweek_info(hosp_reporting, 'date')
    
    # Get list of available locations for selector
    available_locations = sorted(forecasts['location_name'].dropna().unique())
    
    return forecasts, target_admissions, ili_data, weather_data, hosp_reporting, available_locations

# Process all data with caching
forecasts, target_admissions, ili_data, weather_data, hosp_reporting, available_locations = process_all_data(
    forecasts, locations, target_admissions, ili_data, weather_data, hosp_reporting
)

app = st.columns(1)

with app[0]:
    st.title("Influenza Hospitalization Forecast: 2025/2026 Season")
    st.markdown("""
    **Forecast for the 2025/2026 influenza season** (starting epidemic week 40, 2025)
    
    The forecast shows:
    - **Median prediction** (solid line)
    - **50% prediction interval** (darker shaded area) - there is a 50% probability the true value falls within this range
    - **80% prediction interval** (lighter shaded area, shown for 1-2 locations) - there is an 80% probability the true value falls within this range
    - **Observed data** (black circles) - actual reported hospitalizations for the current season
    """)
    
    # Location selector
    selected_locations = st.multiselect(
        "Select locations to display:",
        options=available_locations,
        default=[available_locations[0]] if available_locations else []
    )
    
    if not selected_locations:
        st.warning("Please select at least one location.")
    else:
        # Filter data for selected locations
        forecast_data = forecasts[forecasts['location_name'].isin(selected_locations)].copy()
        
        # Get observed data for 2025/2026 season (include epi weeks 35 onwards for this season)
        # For 2025/2026, we want weeks 35-52 of 2025 and weeks 1-20 of 2026
        observed_2025 = target_admissions[
            (target_admissions['location_name'].isin(selected_locations)) &
            (
                ((target_admissions['epiyear'] == 2025) & (target_admissions['epiweek'] >= 35)) |
                ((target_admissions['epiyear'] == 2026) & (target_admissions['epiweek'] <= 20))
            )
        ].copy()
        
        # Determine which quantiles to use based on number of locations
        show_80pi = len(selected_locations) < 3
        
        # Prepare data for median
        median_data = forecast_data[forecast_data['output_type_id'] == 0.5].copy()
        
        # Prepare data for 50% PI
        pi50_lower = forecast_data[forecast_data['output_type_id'] == 0.25].copy()
        pi50_upper = forecast_data[forecast_data['output_type_id'] == 0.75].copy()
        
        # Merge lower and upper bounds for 50% PI
        pi50_data = pi50_lower.merge(
            pi50_upper[['target_end_date', 'location_name', 'value']], 
            on=['target_end_date', 'location_name'],
            suffixes=('_lower', '_upper')
        )
        
        # Prepare data for 80% PI (if needed)
        if show_80pi:
            pi80_lower = forecast_data[forecast_data['output_type_id'] == 0.1].copy()
            pi80_upper = forecast_data[forecast_data['output_type_id'] == 0.9].copy()
            
            pi80_data = pi80_lower.merge(
                pi80_upper[['target_end_date', 'location_name', 'value']], 
                on=['target_end_date', 'location_name'],
                suffixes=('_lower', '_upper')
            )
        
        # Prepare observed data with proper date column name
        if not observed_2025.empty:
            observed_2025_plot = observed_2025.copy()
            observed_2025_plot['target_end_date'] = observed_2025_plot['date']
        
        # Build Altair chart with layered components
        if show_80pi:
            layers = [
                alt.Chart(pi80_data).mark_area(opacity=0.2).encode(
                    x=alt.X('target_end_date:T', title='Date'),
                    y=alt.Y('value_lower:Q', title='Incident flu hospitalizations'),
                    y2='value_upper:Q',
                    color=alt.Color('location_name:N', title='Location')
                ),
                alt.Chart(pi50_data).mark_area(opacity=0.4).encode(
                    x=alt.X('target_end_date:T', title='Date'),
                    y=alt.Y('value_lower:Q', title='Incident flu hospitalizations'),
                    y2='value_upper:Q',
                    color=alt.Color('location_name:N', legend=None)
                ),
                alt.Chart(median_data).mark_line(size=2).encode(
                    x=alt.X('target_end_date:T', title='Date'),
                    y=alt.Y('value:Q', title='Incident flu hospitalizations'),
                    color=alt.Color('location_name:N', legend=None)
                )
            ]
            
            # Add observed data points if available
            if not observed_2025.empty:
                layers.append(
                    alt.Chart(observed_2025_plot).mark_circle(size=60, color='black').encode(
                        x=alt.X('target_end_date:T', title='Date'),
                        y=alt.Y('value:Q', title='Incident flu hospitalizations'),
                        tooltip=['location_name:N', 'target_end_date:T', 'value:Q']
                    )
                )
            
            chart = alt.layer(*layers).properties(
                width=800,
                height=400,
                title={
                    "text": "2025/2026 Season Forecast",
                    "subtitle": ["Median forecast with 50% PI (darker) and 80% PI (lighter)", 
                                 "Black circles show observed hospitalizations"],
                    "fontSize": 16,
                    "subtitleFontSize": 12,
                    "anchor": "start"
                }
            ).interactive()
        else:
            layers = [
                alt.Chart(pi50_data).mark_area(opacity=0.4).encode(
                    x=alt.X('target_end_date:T', title='Date'),
                    y=alt.Y('value_lower:Q', title='Incident flu hospitalizations'),
                    y2='value_upper:Q',
                    color=alt.Color('location_name:N', title='Location')
                ),
                alt.Chart(median_data).mark_line(size=2).encode(
                    x=alt.X('target_end_date:T', title='Date'),
                    y=alt.Y('value:Q', title='Incident flu hospitalizations'),
                    color=alt.Color('location_name:N', legend=None)
                )
            ]
            
            # Add observed data points if available
            if not observed_2025.empty:
                layers.append(
                    alt.Chart(observed_2025_plot).mark_circle(size=60, color='black').encode(
                        x=alt.X('target_end_date:T', title='Date'),
                        y=alt.Y('value:Q', title='Incident flu hospitalizations'),
                        tooltip=['location_name:N', 'target_end_date:T', 'value:Q']
                    )
                )
            
            chart = alt.layer(*layers).properties(
                width=800,
                height=400,
                title={
                    "text": "2025/2026 Season Forecast",
                    "subtitle": ["Median forecast with 50% prediction interval (shaded area)", 
                                 "Black circles show observed hospitalizations"],
                    "fontSize": 16,
                    "subtitleFontSize": 12,
                    "anchor": "start"
                }
            ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        # Supporting Information Section
        st.header("Supporting Data")
        st.write("Historical context and related indicators for the selected location(s)")
        
        # Filter datasets by selected locations
        target_admissions_filtered = target_admissions[target_admissions['location_name'].isin(selected_locations)].copy()
        ili_data_filtered = ili_data[ili_data['location_name'].isin(selected_locations)].copy()
        weather_data_filtered = weather_data[weather_data['location_name'].isin(selected_locations)].copy()
        hosp_reporting_filtered = hosp_reporting[hosp_reporting['location_name'].isin(selected_locations)].copy()
        
        # Row 1: Hospital Admissions and ILI
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hospital Admissions by Season")
            if not target_admissions_filtered.empty:
                # Remove NA values
                target_admissions_filtered = target_admissions_filtered.dropna(subset=['value'])
                
                hosp_chart = alt.Chart(target_admissions_filtered).mark_line().encode(
                    x=alt.X('weeks_from_start:Q', title='Weeks from Season Start (Week 40)', scale=alt.Scale(domain=[0, 33])),
                    y=alt.Y('value:Q', title='Hospital Admissions'),
                    color=alt.Color('season:N', title='Season'),
                    strokeDash=alt.StrokeDash('location_name:N', title='Location') if len(selected_locations) > 1 else alt.value([0])
                ).properties(
                    height=300
                ).interactive()
                st.altair_chart(hosp_chart, use_container_width=True)
            else:
                st.info("No hospital admissions data available for selected location(s)")
        
        with col2:
            st.subheader("ILI Percent Over Time")
            if not ili_data_filtered.empty:
                ili_chart = alt.Chart(ili_data_filtered).mark_line().encode(
                    x=alt.X('weeks_from_start:Q', title='Weeks from Season Start (Week 40)', scale=alt.Scale(domain=[0, 33])),
                    y=alt.Y('wili:Q', title='Weighted ILI (%)'),
                    color=alt.Color('season:N', title='Season'),
                    strokeDash=alt.StrokeDash('location_name:N', title='Location') if len(selected_locations) > 1 else alt.value([0])
                ).properties(
                    height=300
                ).interactive()
                st.altair_chart(ili_chart, use_container_width=True)
            else:
                st.info("No ILI data available for selected location(s)")
        
        # Row 2: Temperature and Pressure (separate plots)
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Temperature")
            if not weather_data_filtered.empty:
                # Filter to seasons 2020/2021 and forward
                weather_filtered_2020 = weather_data_filtered[
                    weather_data_filtered['season'].apply(lambda x: int(x.split('/')[0]) >= 2020)
                ].copy()
                
                if not weather_filtered_2020.empty:
                    # Average across locations and group by weeks_from_start and season
                    if len(selected_locations) > 1:
                        weather_avg = weather_filtered_2020.groupby(['weeks_from_start', 'season']).agg({
                            'tavg': 'mean'
                        }).reset_index()
                        temp_plot_data = weather_avg
                    else:
                        temp_plot_data = weather_filtered_2020[['weeks_from_start', 'season', 'tavg']]
                    
                    # Create temperature chart
                    temp_chart = alt.Chart(temp_plot_data).mark_line().encode(
                        x=alt.X('weeks_from_start:Q', title='Weeks from Season Start (Week 40)', scale=alt.Scale(domain=[0, 33])),
                        y=alt.Y('tavg:Q', title='Temperature (°C)'),
                        color=alt.Color('season:N', title='Season')
                    ).properties(
                        height=300
                    ).interactive()
                    st.altair_chart(temp_chart, use_container_width=True)
                else:
                    st.info("No weather data available for selected location(s) from 2020 onwards")
            else:
                st.info("No weather data available for selected location(s)")
        
        with col4:
            st.subheader("Pressure")
            if not weather_data_filtered.empty:
                # Filter to seasons 2020/2021 and forward
                weather_filtered_2020 = weather_data_filtered[
                    weather_data_filtered['season'].apply(lambda x: int(x.split('/')[0]) >= 2020)
                ].copy()
                
                if not weather_filtered_2020.empty:
                    # Average across locations and group by weeks_from_start and season
                    if len(selected_locations) > 1:
                        weather_avg = weather_filtered_2020.groupby(['weeks_from_start', 'season']).agg({
                            'pres': 'mean'
                        }).reset_index()
                        pres_plot_data = weather_avg
                    else:
                        pres_plot_data = weather_filtered_2020[['weeks_from_start', 'season', 'pres']]
                    
                    # Create pressure chart
                    pres_chart = alt.Chart(pres_plot_data).mark_line().encode(
                        x=alt.X('weeks_from_start:Q', title='Weeks from Season Start (Week 40)', scale=alt.Scale(domain=[0, 33])),
                        y=alt.Y('pres:Q', title='Pressure (hPa)', scale=alt.Scale(zero=False)),
                        color=alt.Color('season:N', title='Season')
                    ).properties(
                        height=300
                    ).interactive()
                    st.altair_chart(pres_chart, use_container_width=True)
                else:
                    st.info("No weather data available for selected location(s) from 2020 onwards")
            else:
                st.info("No weather data available for selected location(s)")
        
        # Row 3: Hospital Reporting Rate
        st.subheader("Hospital Reporting Rate")
        if not hosp_reporting_filtered.empty:
            hosp_reporting_chart = alt.Chart(hosp_reporting_filtered).mark_line().encode(
                x=alt.X('weeks_from_start:Q', title='Weeks from Season Start (Week 40)', scale=alt.Scale(domain=[0, 33])),
                y=alt.Y('pct_hosp:Q', title='% Hospitals Reporting', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('season:N', title='Season'),
                strokeDash=alt.StrokeDash('location_name:N', title='Location') if len(selected_locations) > 1 else alt.value([0])
            ).properties(
                height=300
            ).interactive()
            st.altair_chart(hosp_reporting_chart, use_container_width=True)
        else:
            st.info("No hospital reporting data available for selected location(s)")

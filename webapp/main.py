#mcandrew

import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import boto3

# AWS S3 configuration
bucket_name = 'flu-forecast-inc-nodata'
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# Load forecast data from S3
response = s3.get_object(Bucket=bucket_name, Key='forecasts__2025-11-01.csv')
forecasts = pd.read_csv(response['Body'])

# Load locations data from S3
response = s3.get_object(Bucket=bucket_name, Key="locations.csv")
locations = pd.read_csv(response['Body'])

# Filter for reference_date 2025-11-01
forecasts = forecasts[forecasts['reference_date'] == '2025-11-01'].copy()

# Convert location to string for merging
forecasts['location'] = forecasts['location'].astype(str)
locations['location'] = locations['location'].astype(str)
def format_location(row):
    if row['location'] == "US":
        return "US"
    else:
        return "{:02d}".format(int(row['location']))
forecasts['location'] = forecasts.apply(lambda x: format_location(x), axis=1)
locations['location'] = locations.apply(lambda x: format_location(x), axis=1)

# Merge to get location names
forecasts = forecasts.merge(locations[['location', 'location_name']], on='location', how='left')

# Get list of available locations for selector
available_locations = sorted(forecasts['location_name'].dropna().unique())

app = st.columns(1)

with app[0]:
    st.title("Influenza Hospitalization Forecast")
    st.write("This is a forecast of influenza hospitalizations for the current season.")
    st.write("The forecast is based on the historical data and the current season data.")
    
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
        
        # Build Altair chart with layered components
        if show_80pi:
            chart = (
                alt.layer(
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
                ).properties(
                    width=800,
                    height=400
                ).interactive()
            )
        else:
            chart = (
                alt.layer(
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
                ).properties(
                    width=800,
                    height=400
                ).interactive()
            )
        
        st.altair_chart(chart, use_container_width=True)

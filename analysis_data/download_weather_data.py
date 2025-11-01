#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from meteostat import Point, Daily
from datetime import datetime

if __name__ == "__main__":

    state_to_fips = {
        "Alabama": "01",
        "Alaska": "02",
        "Arizona": "04",
        "Arkansas": "05",
        "California": "06",
        "Colorado": "08",
        "Connecticut": "09",
        "Delaware": "10",
        "Florida": "12",
        "Georgia": "13",
        "Hawaii": "15",
        "Idaho": "16",
        "Illinois": "17",
        "Indiana": "18",
        "Iowa": "19",
        "Kansas": "20",
        "Kentucky": "21",
        "Louisiana": "22",
        "Maine": "23",
        "Maryland": "24",
        "Massachusetts": "25",
        "Michigan": "26",
        "Minnesota": "27",
        "Mississippi": "28",
        "Missouri": "29",
        "Montana": "30",
        "Nebraska": "31",
        "Nevada": "32",
        "New Hampshire": "33",
        "New Jersey": "34",
        "New Mexico": "35",
        "New York": "36",
        "North Carolina": "37",
        "North Dakota": "38",
        "Ohio": "39",
        "Oklahoma": "40",
        "Oregon": "41",
        "Pennsylvania": "42",
        "Rhode Island": "44",
        "South Carolina": "45",
        "South Dakota": "46",
        "Tennessee": "47",
        "Texas": "48",
        "Utah": "49",
        "Vermont": "50",
        "Virginia": "51",
        "Washington": "53",
        "West Virginia": "54",
        "Wisconsin": "55",
        "Wyoming": "56",
        "District of Columbia": "11",
        "Puerto Rico": "72"
    }

    # Largest 3 cities for each U.S. state with coordinates
    locations = {
        "Alabama": [
            ("Birmingham", Point(33.5186, -86.8104)),
            ("Montgomery", Point(32.3668, -86.3000)),
            ("Huntsville", Point(34.7304, -86.5861)),
        ],
        "Alaska": [
            ("Anchorage", Point(61.2181, -149.9003)),
            ("Fairbanks", Point(64.8378, -147.7164)),
            ("Juneau", Point(58.3019, -134.4197)),
        ],
        "Arizona": [
            ("Phoenix", Point(33.4484, -112.0740)),
            ("Tucson", Point(32.2226, -110.9747)),
            ("Mesa", Point(33.4152, -111.8315)),
        ],
        "Arkansas": [
            ("Little Rock", Point(34.7465, -92.2896)),
            ("Fort Smith", Point(35.3859, -94.3985)),
            ("Fayetteville", Point(36.0822, -94.1719)),
        ],
        "Puerto Rico": [
            ("San Juan", Point(18.4663, -66.1181)),
            ("Bayamón", Point(18.3962, -66.1558)),
            ("Carolina", Point(18.3962, -66.1558)),
        ],
        "District of Columbia": [
            ("Washington", Point(38.8936, -77.0321)),
            ("Alexandria", Point(38.8048, -77.0469)),
            ("Arlington", Point(38.8811, -77.0247)),
        ],
        "California": [
            ("Los Angeles", Point(34.0522, -118.2437)),
            ("San Diego", Point(32.7157, -117.1611)),
            ("San Jose", Point(37.3382, -121.8863)),
        ],
        "Colorado": [
            ("Denver", Point(39.7392, -104.9903)),
            ("Colorado Springs", Point(38.8339, -104.8214)),
            ("Aurora", Point(39.7294, -104.8319)),
        ],
        "Connecticut": [
            ("Bridgeport", Point(41.1792, -73.1894)),
            ("New Haven", Point(41.3083, -72.9279)),
            ("Stamford", Point(41.0534, -73.5387)),
        ],
        "Delaware": [
            ("Wilmington", Point(39.7459, -75.5466)),
            ("Dover", Point(39.1582, -75.5244)),
            ("Newark", Point(39.6837, -75.7497)),
        ],
        "Florida": [
            ("Jacksonville", Point(30.3322, -81.6557)),
            ("Miami", Point(25.7617, -80.1918)),
            ("Tampa", Point(27.9506, -82.4572)),
        ],
        "Georgia": [
            ("Atlanta", Point(33.7490, -84.3880)),
            ("Augusta", Point(33.4735, -82.0105)),
            ("Columbus", Point(32.4608, -84.9877)),
        ],
        "Hawaii": [
            ("Honolulu", Point(21.3069, -157.8583)),
            ("Pearl City", Point(21.3972, -157.9752)),
            ("Hilo", Point(19.7074, -155.0859)),
        ],
        "Idaho": [
            ("Boise", Point(43.6150, -116.2023)),
            ("Meridian", Point(43.6121, -116.3915)),
            ("Nampa", Point(43.5407, -116.5635)),
        ],
        "Illinois": [
            ("Chicago", Point(41.8781, -87.6298)),
            ("Aurora", Point(41.7606, -88.3201)),
            ("Naperville", Point(41.7508, -88.1535)),
        ],
        "Indiana": [
            ("Indianapolis", Point(39.7684, -86.1581)),
            ("Fort Wayne", Point(41.0793, -85.1394)),
            ("Evansville", Point(37.9716, -87.5711)),
        ],
        "Iowa": [
            ("Des Moines", Point(41.5868, -93.6250)),
            ("Cedar Rapids", Point(42.0083, -91.6441)),
            ("Davenport", Point(41.5236, -90.5776)),
        ],
        "Kansas": [
            ("Wichita", Point(37.6872, -97.3301)),
            ("Overland Park", Point(38.9822, -94.6708)),
            ("Kansas City", Point(39.1142, -94.6275)),
        ],
        "Kentucky": [
            ("Louisville", Point(38.2527, -85.7585)),
            ("Lexington", Point(38.0406, -84.5037)),
            ("Bowling Green", Point(36.9685, -86.4808)),
        ],
        "Louisiana": [
            ("New Orleans", Point(29.9511, -90.0715)),
            ("Baton Rouge", Point(30.4515, -91.1871)),
            ("Shreveport", Point(32.5252, -93.7502)),
        ],
        "Maine": [
            ("Portland", Point(43.6615, -70.2553)),
            ("Lewiston", Point(44.1003, -70.2148)),
            ("Bangor", Point(44.8012, -68.7778)),
        ],
        "Maryland": [
            ("Baltimore", Point(39.2904, -76.6122)),
            ("Frederick", Point(39.4143, -77.4105)),
            ("Rockville", Point(39.0839, -77.1528)),
        ],
        "Massachusetts": [
            ("Boston", Point(42.3601, -71.0589)),
            ("Worcester", Point(42.2626, -71.8023)),
            ("Springfield", Point(42.1015, -72.5898)),
        ],
        "Michigan": [
            ("Detroit", Point(42.3314, -83.0458)),
            ("Grand Rapids", Point(42.9634, -85.6681)),
            ("Warren", Point(42.5145, -83.0147)),
        ],
        "Minnesota": [
            ("Minneapolis", Point(44.9778, -93.2650)),
            ("Saint Paul", Point(44.9537, -93.0900)),
            ("Rochester", Point(44.0121, -92.4802)),
        ],
        "Mississippi": [
            ("Jackson", Point(32.2988, -90.1848)),
            ("Gulfport", Point(30.3674, -89.0928)),
            ("Southaven", Point(34.9889, -90.0126)),
        ],
        "Missouri": [
            ("Kansas City", Point(39.0997, -94.5786)),
            ("Saint Louis", Point(38.6270, -90.1994)),
            ("Springfield", Point(37.2089, -93.2923)),
        ],
        "Montana": [
            ("Billings", Point(45.7833, -108.5007)),
            ("Missoula", Point(46.8721, -113.9940)),
            ("Great Falls", Point(47.5052, -111.3008)),
        ],
        "Nebraska": [
            ("Omaha", Point(41.2565, -95.9345)),
            ("Lincoln", Point(40.8136, -96.7026)),
            ("Bellevue", Point(41.1544, -95.9146)),
        ],
        "Nevada": [
            ("Las Vegas", Point(36.1699, -115.1398)),
            ("Henderson", Point(36.0395, -114.9817)),
            ("Reno", Point(39.5296, -119.8138)),
        ],
        "New Hampshire": [
            ("Manchester", Point(42.9956, -71.4548)),
            ("Nashua", Point(42.7654, -71.4676)),
            ("Concord", Point(43.2081, -71.5376)),
        ],
        "New Jersey": [
            ("Newark", Point(40.7357, -74.1724)),
            ("Jersey City", Point(40.7178, -74.0431)),
            ("Paterson", Point(40.9168, -74.1718)),
        ],
        "New Mexico": [
            ("Albuquerque", Point(35.0844, -106.6504)),
            ("Las Cruces", Point(32.3199, -106.7637)),
            ("Rio Rancho", Point(35.2334, -106.6645)),
        ],
        "New York": [
            ("New York City", Point(40.7128, -74.0060)),
            ("Buffalo", Point(42.8864, -78.8784)),
            ("Rochester", Point(43.1566, -77.6088)),
        ],
        "North Carolina": [
            ("Charlotte", Point(35.2271, -80.8431)),
            ("Raleigh", Point(35.7796, -78.6382)),
            ("Greensboro", Point(36.0726, -79.7910)),
        ],
        "North Dakota": [
            ("Fargo", Point(46.8772, -96.7898)),
            ("Bismarck", Point(46.8083, -100.7837)),
            ("Grand Forks", Point(47.9253, -97.0329)),
        ],
        "Ohio": [
            ("Columbus", Point(39.9612, -82.9988)),
            ("Cleveland", Point(41.4993, -81.6944)),
            ("Cincinnati", Point(39.1031, -84.5120)),
        ],
        "Oklahoma": [
            ("Oklahoma City", Point(35.4676, -97.5164)),
            ("Tulsa", Point(36.1539, -95.9928)),
            ("Norman", Point(35.2226, -97.4395)),
        ],
        "Oregon": [
            ("Portland", Point(45.5051, -122.6750)),
            ("Eugene", Point(44.0521, -123.0868)),
            ("Salem", Point(44.9429, -123.0351)),
        ],
        "Pennsylvania": [
            ("Philadelphia", Point(39.9526, -75.1652)),
            ("Pittsburgh", Point(40.4406, -79.9959)),
            ("Allentown", Point(40.6084, -75.4902)),
        ],
        "Rhode Island": [
            ("Providence", Point(41.8240, -71.4128)),
            ("Cranston", Point(41.7798, -71.4373)),
            ("Warwick", Point(41.7001, -71.4162)),
        ],
        "South Carolina": [
            ("Columbia", Point(34.0007, -81.0348)),
            ("Charleston", Point(32.7765, -79.9311)),
            ("North Charleston", Point(32.8546, -79.9748)),
        ],
        "South Dakota": [
            ("Sioux Falls", Point(43.5473, -96.7283)),
            ("Rapid City", Point(44.0805, -103.2310)),
            ("Aberdeen", Point(45.4647, -98.4865)),
        ],
        "Tennessee": [
            ("Nashville", Point(36.1627, -86.7816)),
            ("Memphis", Point(35.1495, -90.0490)),
            ("Knoxville", Point(35.9606, -83.9207)),
        ],
        "Texas": [
            ("Houston", Point(29.7604, -95.3698)),
            ("San Antonio", Point(29.4241, -98.4936)),
            ("Dallas", Point(32.7767, -96.7970)),
        ],
        "Utah": [
            ("Salt Lake City", Point(40.7608, -111.8910)),
            ("West Valley City", Point(40.6916, -112.0010)),
            ("Provo", Point(40.2338, -111.6585)),
        ],
        "Vermont": [
            ("Burlington", Point(44.4759, -73.2121)),
            ("South Burlington", Point(44.4665, -73.1712)),
            ("Rutland", Point(43.6106, -72.9726)),
        ],
        "Virginia": [
            ("Virginia Beach", Point(36.8529, -75.9780)),
            ("Norfolk", Point(36.8508, -76.2859)),
            ("Chesapeake", Point(36.7682, -76.2875)),
        ],
        "Washington": [
            ("Seattle", Point(47.6062, -122.3321)),
            ("Spokane", Point(47.6588, -117.4260)),
            ("Tacoma", Point(47.2529, -122.4443)),
        ],
        "West Virginia": [
            ("Charleston", Point(38.3498, -81.6326)),
            ("Huntington", Point(38.4192, -82.4452)),
            ("Morgantown", Point(39.6295, -79.9559)),
        ],
        "Wisconsin": [
            ("Milwaukee", Point(43.0389, -87.9065)),
            ("Madison", Point(43.0731, -89.4012)),
            ("Green Bay", Point(44.5192, -88.0198)),
        ],
        "Wyoming": [
            ("Cheyenne", Point(41.1400, -104.8202)),
            ("Casper", Point(42.8501, -106.3252)),
            ("Laramie", Point(41.3114, -105.5911)),
        ]}

    # Define the date range
    start_date = datetime(2010, 1, 1)
    #end_date   = datetime(2025, 5, 22)
    end_date = datetime.now()

    # Initialize a DataFrame to store results
    all_data = pd.DataFrame()

    # Fetch and process data for each location
    for state,city_data in locations.items():
        for (location_name), latlong in city_data:
            # Fetch daily data for the location
            data = Daily(latlong, start_date, end_date).fetch()

            # Add a 'Week' column
            data['Week'] = data.index#.to_period('W')

            # Calculate weekly average temperature
            weekly_data = data.groupby('Week')[['tavg',"pres"]].mean()

            weekly_data["location_name"] = location_name
            weekly_data["state"]      = state

            # Merge into the main DataFrame
            all_data = pd.concat([all_data,weekly_data])

    all_data = all_data.reset_index()

    all_data["FIPS"] = all_data.state.replace(state_to_fips)
    
    def avg_weather(x):
        return pd.Series({"tavg":np.nanmean(x.tavg), "pres":np.nanmean(x.pres)})
    all_data_mean         = all_data.groupby(["state","FIPS","Week"]).apply(avg_weather).reset_index()

    all_data_mean         = all_data_mean.rename(columns = {"FIPS":"location"})

    all_data      = all_data[["FIPS","state","Week","tavg","pres"]]
    all_data      = all_data.rename(columns = {"state":"location_name","FIPS":"location"})
    all_data_mean = all_data_mean.rename(columns = {"state":"location_name","FIPS":"location"})

    #--add in epi time data
    def from_week_to_epiweek(row):
        from epiweeks import Week
        week    = row.Week
        eweek   = Week.fromdate( datetime(week.year,week.month,week.day) )

        row["year"]    = week.year
        row["epiweek"] = eweek.cdcformat()
        row["week"]    = eweek.week
        row["enddate"] = eweek.enddate()

        return row

    times = all_data_mean[["Week"]].drop_duplicates()
    times = times.apply(from_week_to_epiweek, 1)

    all_data_mean = all_data_mean.merge(times, on = ["Week"])
    
    #--from daily to weekl
    def avg_temp(x):
        x["tavg"]    = np.nanmean(x["tavg"])
        x["pavg"]    = np.nanmean(x["pres"])
        return x.iloc[-1]
    end_of_week_data = all_data_mean.groupby(["enddate","location","location_name"]).apply( avg_temp ).reset_index(drop=True)

    #--US data is an average
    def average_weather(x):
        return pd.Series({"tavg":np.nanmean(x["tavg"]), "pavg":np.nanmean(x["pavg"]), "pres":np.nanmean(x["pres"]) } )
    us = end_of_week_data.groupby(["Week","year","week","epiweek","enddate"]).apply(average_weather).reset_index()
    us["location"] = "US"
    us["location_name"] = "US"

    end_of_week_data = pd.concat([end_of_week_data, us])

    # Optionally save to a CSV
    end_of_week_data.to_csv('./analysis_data/weekly_weather_data.csv', index=False)

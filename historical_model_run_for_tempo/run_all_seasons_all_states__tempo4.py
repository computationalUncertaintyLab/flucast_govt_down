#mcandrew

import sys
sys.path.append('./model/tempo/')
from tempo_model import tempo_model4

import numpy as np
import pandas as pd

from pathlib import Path

from joblib import Parallel, delayed

def from_time_to_season(x, yrstop):

    yr,week = x.MMWRYR, x.MMWRWK

    if yr==yrstop and week>=40:
        season = "-1"
        return season

    if week>20 and week<40:
        season="-1"
    else:
        if week>=40:
            season = "{:d}/{:d}".format( yr, yr+1)
        else:
            season = "{:d}/{:d}".format( yr-1, yr)
    return season

def format_counts(d,interp=False):
    weeks = list(np.arange(40,53+1)) + list(np.arange(1,20+1)) 

    for week in weeks:
        if week not in d.columns:
            d[ week ] = np.nan
    d = d[weeks]

    if interp:
        d_counts = []
        for row in d.to_numpy():
            d_counts.append(interpolate_nans(row))
        d_counts = np.array(d_counts)
    else:
        d_counts = d.to_numpy()
    return d_counts



def add_time_data(row):
    from epiweeks import Week
    from datetime import datetime

    epiweek = Week.fromdate( datetime.strptime(row.Week,"%Y-%m-%d"))
    row["MMWRYR"] = epiweek.year
    row["MMWRWK"] = epiweek.week
    row["season"] = "{:d}/{:d}".format(epiweek.year,epiweek.year+1) if epiweek.week >=40 else "{:d}/{:d}".format(epiweek.year-1,epiweek.year)

    return row

def interpolate_nans(array):
    """
    Linearly interpolates NaN values in a 1D NumPy array.
    For leading/trailing NaNs, it performs forward/backward filling.
    """
    nans = np.isnan(array)
    # Create an array of indices for the original array
    x = np.arange(len(array))
    
    # Use np.interp to fill NaNs
    # x=x[nans]: Indices where NaNs are present (where we want to interpolate)
    # xp=x[~nans]: Indices where non-NaN values are present (known points)
    # fp=array[~nans]: Values at the non-NaN indices (known values)
    array[nans] = np.interp(x=x[nans], xp=x[~nans], fp=array[~nans])
    
    return array

if __name__ == "__main__":

    THIS_SEASON = "2025/2026"
    
    #--data set of populations (contains all FIPS)
    pops                = pd.read_csv("./data/locations.csv")
    
    #--incident hospitalizations dataset
    inc_hosps           = pd.read_csv("./data/target-data/target-hospital-admissions.csv")

    pct_hosps_reporting           = pd.read_csv("./analysis_data/pct_hospital_reporting.csv")
    pct_hosps_reporting["season"] = pct_hosps_reporting.apply(lambda row:from_time_to_season(row,2026), 1)
    pct_hosps_reporting           = pct_hosps_reporting.loc[pct_hosps_reporting.season!='-1']
    
    
    #--subset by only information after 09-01
    inc_hosps           = inc_hosps.loc[ (inc_hosps["date"]>="2021-10-09")  ]

    #--ILI data
    ili_data            = pd.read_csv("./analysis_data/ili_data_all_states_2021_present__formatted.csv")
    ili_data["week"]    = [ int(str(x)[-2:]) for x in ili_data.epiweek]

    #--lab data
    lab_data            = pd.read_csv("./analysis_data/clinical_and_public_lab_data__formatted.csv")

    ili_augmented       = lab_data.merge(ili_data, on = ["state","epiweek","year","week"] )
    ili_augmented["region"] = [ "US" if x == 'National' else x for x in ili_augmented.region.values]
    
    ili_augmented = ili_augmented.merge( inc_hosps[["location","location_name"]].drop_duplicates(), left_on =["region"], right_on=["location_name"] )


    #--COVARIATE INFORMATION--------------------------------------------------------------
    weather_data        = pd.read_csv("./analysis_data/weekly_weather_data.csv")
    weather_data        = weather_data.loc[weather_data.year>=2021]

    time_data = weather_data[["Week"]].drop_duplicates()
    time_data = time_data.apply( add_time_data,1 )

    weather_data = weather_data.merge(time_data, on = ["Week"])
    weather_data = weather_data.loc[weather_data.season!="2009/2010"]

    inc_hosps = inc_hosps.merge(time_data           , left_on = ["date"], right_on = ["Week"] )

    ili_augmented = ili_augmented.drop(columns = ["season"])


    ili_augmented  = ili_augmented.merge(time_data  , left_on=["year","week"], right_on=["MMWRYR","MMWRWK"] )

    seasons = ["2021/2022","2022/2023","2023/2024","2024/2025"]
    weather_data = weather_data.loc[weather_data.season.isin(seasons)]
    ili_augmented = ili_augmented.loc[ili_augmented.season.isin(seasons)]
    pct_hosps_reporting = pct_hosps_reporting.loc[pct_hosps_reporting.season.isin(seasons)]

    # Weather covariates processing
    def standardize(x):
        from scipy.ndimage import gaussian_filter1d
        """Standardize and smooth weather data."""
        X = (x - np.nanmin(x, 1).reshape(-1, 1)) / ( np.nanmax(x, 1).reshape(-1, 1) - np.nanmin(x, 1).reshape(-1, 1)  )

        #--if a row is all nan thats because the min and max were equal. In other words, the pct reportign never changed.
        #--in this case we set that row to zero. 
        for row in np.where(np.all(np.isnan(X),1)):
            X.iloc[row] = 0.
        X = X.to_numpy()

        # Apply Gaussian smoothing
        for n, row in enumerate(X):
            X[n,:] = gaussian_filter1d(row, 2)
        return X

    def process_weather_variable(data,variable_name):
        """Process weather variable (temp or pressure) with interpolation and standardization."""
        data = pd.pivot_table(
            index   = "season", 
            columns = ["week"], 
            values  = variable_name, 
            data    = data,
            dropna  = False
        )

        if 53 not in data.columns:
            data[53] = np.nan


        # Select flu season weeks (40-52, 1-20)
        weeks = list(np.arange(40, 53 + 1)) + list(np.arange(1, 20 + 1))
        data  = data[weeks]

        # Interpolate across weeks and seasons
        data = data.interpolate(axis=1)
        data = data.interpolate(axis=0, limit_direction="both")

        return standardize(data)

    # Process temperature and pressure
    
   
    #------
    from_season_to_number = { season:n for n,season in enumerate(sorted(inc_hosps.season.unique())) }
   
    def build_parameter_data( location,  subset, weather_data, pct_hosps_reporting,ili_augmented ):
       
        import os 
        fstring = "./historical_model_run_for_tempo/arxiv__tempo4/params_{:s}.csv".format(location)
        if os.path.exists(fstring):
            return 
        print(location)
        
        param_data = {"location":[],"season":[],"param_type":[], "param1":[],"param2":[],"value":[]}

        #--subset all data to specific state
        temp_data_centered = process_weather_variable(weather_data.loc[weather_data.location==location],"tavg")
        pres_data_centered = process_weather_variable(weather_data.loc[weather_data.location==location],"pavg")
        pct_hosps_centered = process_weather_variable(pct_hosps_reporting.rename(columns={"MMWRWK":"week"}).loc[pct_hosps_reporting.location==location ],"pct_hosp")
        ili_vals_centered  = process_weather_variable(ili_augmented.loc[ili_augmented.location==location],"ili")

        state_ili = ili_augmented.loc[(ili_augmented.location==location) ].drop_duplicates()
        state_ili = state_ili.loc[(state_ili.MMWRWK>=40) | (state_ili.MMWRWK <=20)]
        
        X = np.stack([temp_data_centered, pres_data_centered,pct_hosps_centered,ili_vals_centered], axis=-1)

        #season_number = from_season_to_number[season]

        # weeks_needed = list(np.arange(40,52+1))  + list(np.arange(1,20+1))
        # for week in weeks_needed:
        #     if week not in state_ili.week.values:
        #         if week >=40:
        #             MMWRYR = min(state_ili.MMWRYR)
        #         else:
        #             MMWRYR = max(state_ili.MMWRYR)
                
        #         d_ = pd.DataFrame({"week":[week],"MMWRWK":[week], "MMWRYR":[MMWRYR] })
        #         state_ili = pd.concat([state_ili, d_])

        # weeks_needed = pd.DataFrame({"MMWRWK":weeks_needed})
        # state_ili = weeks_needed.merge(state_ili, on = ["MMWRWK"])
        # subset    = weeks_needed.merge(subset, on = ["MMWRWK"])

        def format_counts(d,interp=False):
            d.columns = [y for x,y in d.columns]
            
            weeks = list(np.arange(40,53+1)) + list(np.arange(1,20+1)) 

            for week in weeks:
                if week not in d.columns:
                    d[ week ] = np.nan
            d = d[weeks]

            if interp:
                d_counts = []
                for row in d.to_numpy():
                    d_counts.append(interpolate_nans(row))
                d_counts = np.array(d_counts)
            else:
                d_counts = d.to_numpy()
            return d_counts
        N               = pd.pivot_table(index=["season"],columns = ["week"],values=["num_patients"], data = state_ili)
        ttl_flu_         = pd.pivot_table(index=["season"],columns = ["MMWRWK"],values=["value"], data = subset)

        N       = format_counts(N, interp=True)
        ttl_flu = format_counts(ttl_flu_)
        
        # #--need to add a "53rd"week if one does not exist and fill it with NA
        # if 53 not in subset.MMWRWK.values:
        #     part_one = subset.loc[ (subset.MMWRWK>=40) & (subset.MMWRWK<=52), "value" ].values
        #     part_two = subset.loc[ (subset.MMWRWK>=1) & (subset.MMWRWK<=20) , "value" ].values
        #     ttl_flu  = np.append( np.append( part_one, np.array([np.nan]) ), part_two)

        #     part_one = state_ili.loc[ (state_ili.week>=40) & (state_ili.week<=52), "num_patients" ].values
        #     part_two = state_ili.loc[ (state_ili.week>=1) & (state_ili.week<=20) , "num_patients" ].values
        #     N         = np.append( np.append( part_one, np.array([np.nan]) ), part_two)

        # else:
        #     ttl_flu   = subset["value"].values
        #     N         = state_ili.num_patients.values

        # ttl_flu = interpolate_nans(ttl_flu)
 
        # N    = interpolate_nans(N)

        import jax 
        base_key    = jax.random.PRNGKey(20200320)
        worker_key  = jax.random.fold_in(base_key, 1)
 
        model = tempo_model4( y = (ttl_flu+1.), X = X, N = N, key = worker_key ).fit_past_seasons()

        #--load up parameter data--
        for season_number,season in enumerate(ttl_flu_.index):
            prior_matrix = np.array([])
            for param in ["delta_season","M_season", "B_season","nu_season","Q_season","rho_season","sigma_ar_season"] :
                prior_vector = np.array(model[param][:,season_number])
                prior_vector = prior_vector.flatten()
                prior_matrix = np.vstack([prior_matrix, prior_vector]) if prior_matrix.size > 0 else prior_vector

            #--treat F as a set of ncov variables F1,F2, etc
            for n,prior_vector in enumerate(model["F_season"][:,season_number].T):
                prior_vector = prior_vector.flatten()
                prior_matrix = np.vstack([prior_matrix, prior_vector]) if prior_matrix.size > 0 else prior_vector

            mu  = prior_matrix.mean(1)
            cov = (prior_matrix - mu.reshape(-1,1))
            cov = (cov @ cov.T) / prior_matrix.shape[-1]

            #--unroll data into dict
            param_names = ["delta_season","M_season", "B_season","nu_season","Q_season","rho_season","sigma_ar_season","F1_season","F2_season","F3_season","F4_season"]
            for param, param_mu in zip(param_names,mu):
                param_data["location"].append(location)
                param_data["season"].append(season)
                param_data["param_type"].append("mu")
                param_data["param1"].append(param)
                param_data["param2"].append(param)
                param_data["value"].append(param_mu)

            for row,param1 in zip(cov,param_names):
                for param_cov,param2 in zip(row,param_names):
                    param_data["location"].append(location)
                    param_data["season"].append(season)
                    param_data["param_type"].append("cov")
                    param_data["param1"].append(param1)
                    param_data["param2"].append(param2)
                    param_data["value"].append(param_cov)

        param_data = pd.DataFrame(param_data)
        param_data.to_csv(fstring)


    def tryit(location,subset,weather_data, pct_hosps_reporting,ili_augmented):
        try:
            build_parameter_data(location,subset,weather_data, pct_hosps_reporting,ili_augmented)
        except:
           print("Fail")
           print(location)

    # for (location), subset in inc_hosps.groupby("location"):
    #     build_parameter_data(location,subset,weather_data, pct_hosps_reporting,ili_augmented)
    #     break
    

    #     if  (location=="36"):
    #         continue
    #     print(location)
    #     print(season)
    #     build_parameter_data(location,season,subset,weather_data, pct_hosps_reporting,ili_augmented)

    Parallel(n_jobs=20)( delayed(tryit)(location,subset,weather_data, pct_hosps_reporting,ili_augmented) for (location), subset in inc_hosps.groupby("location") )

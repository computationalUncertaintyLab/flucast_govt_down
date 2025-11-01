#mcandrew

import sys
sys.path.append('./model/tempo/')
from tempo_model import tempo_model4

import numpy as np
import pandas as pd

from pathlib import Path

from epiweeks import Week

from joblib import Parallel, delayed

import os

def from_time_to_season(x, yrstop):

    yr,week = x.MMWRYR, x.MMWRWK

    if yr==yrstop and week>=35:
        season = "-1"
        return season

    if week>20 and week<35:
        season="-1"
    else:
        if week>=40:
            season = "{:d}/{:d}".format( yr, yr+1)
        else:
            season = "{:d}/{:d}".format( yr-1, yr)
    return season

def add_time_data(row):
    from epiweeks import Week
    from datetime import datetime

    epiweek = Week.fromdate( datetime.strptime(row.Week,"%Y-%m-%d"))
    row["MMWRYR"] = epiweek.year
    row["MMWRWK"] = epiweek.week
    row["season"] = "{:d}/{:d}".format(epiweek.year,epiweek.year+1) if epiweek.week >=35 else "{:d}/{:d}".format(epiweek.year-1,epiweek.year)

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

    pct_hosps_reporting = pd.read_csv("./analysis_data/pct_hospital_reporting.csv")
    pct_hosps_reporting["season"] = pct_hosps_reporting.apply(lambda row:from_time_to_season(row,2026), 1)
    pct_hosps_reporting = pct_hosps_reporting.loc[pct_hosps_reporting.season!='-1']
    
    
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

    seasons = ["2021/2022","2022/2023","2023/2024","2024/2025", "2025/2026"]
    weather_data        = weather_data.loc[weather_data.season.isin(seasons)]
    ili_augmented       = ili_augmented.loc[ili_augmented.season.isin(seasons)]
    pct_hosps_reporting = pct_hosps_reporting.loc[pct_hosps_reporting.season.isin(seasons)]

    seasons             = ["2025/2026"]
    past_inc_hosps      = inc_hosps.loc[~inc_hosps.season.isin(seasons)] 
    inc_hosps           = inc_hosps.loc[inc_hosps.season.isin(seasons)] 
    
    # Weather covariates processing
    def standardize(x):
        from scipy.ndimage import gaussian_filter1d
        """Standardize and smooth weather data."""
        X = (x - np.nanmin(x, 1).reshape(-1, 1)) / (np.nanmax(x, 1).reshape(-1, 1) - np.nanmin(x, 1).reshape(-1, 1))
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
        weeks = list(np.arange(35, 53 + 1)) + list(np.arange(1, 20 + 1))
        data  = data[weeks]

        # Interpolate across weeks and seasons
        data = data.interpolate(axis=0)
        data = data.interpolate(axis=1, limit_direction="both")

        return standardize(data)

    #------
    from_season_to_number = { season:n for n,season in enumerate(sorted(inc_hosps.season.unique())) }

    all_params = pd.read_csv("./historical_model_run_for_tempo/all_past_param_estimates__tempo4.csv")

    def format(x):
        if x=="US":
            return x
        return "{:02d}".format(int(x))
    all_params["location"] = [format(x) for x in all_params.location.values]

    def forecast( location, season, subset, weather_data, pct_hosps_reporting,ili_augmented ):
        print(location)
        param_data = {"location":[],"season":[],"param_type":[], "param1":[],"param2":[],"value":[]}

        #--subset all data to specific state
        season = "2025/2026"
    
        thisweek = Week.thisweek().enddate().strftime("%Y-%m-%d")
        forecast_file = "./forecasts/forecast_{:02d}_{:s}.csv".format( int(location),thisweek)
        if os.path.exists(forecast_file):
            return
        
        temp_data_centered = process_weather_variable(weather_data.loc[weather_data.location==location],"tavg")
        pres_data_centered = process_weather_variable(weather_data.loc[weather_data.location==location],"pavg")
        pct_hosps_centered = process_weather_variable(pct_hosps_reporting.rename(columns={"MMWRWK":"week"}).loc[pct_hosps_reporting.location==location ],"pct_hosp")
        ili_vals_centered  = process_weather_variable(ili_augmented.loc[ili_augmented.location==location],"ili")

        state_ili = ili_augmented.loc[(ili_augmented.location==location) & (ili_augmented.season==season)].drop_duplicates()
        state_ili = state_ili.loc[(state_ili.MMWRWK>=35) | (state_ili.MMWRWK <=20)]
        
        past_inc_hosps_state = past_inc_hosps.loc[(past_inc_hosps.location==location) ]
        past_inc_hosps_MMWR20 = past_inc_hosps_state.loc[past_inc_hosps_state.MMWRWK==20]
        mean_past_inc_hosps_MMWR20 = np.nanmean(past_inc_hosps_MMWR20.value.values)
        sd_past_inc_hosps_MMWR20   = np.nanstd(past_inc_hosps_MMWR20.value.values)

        X = np.stack([temp_data_centered[-1,:], pres_data_centered[-1,:],pct_hosps_centered[-1,:],ili_vals_centered[-1,:]], axis=-1)

        weeks_needed = list(np.arange(35,52+1))  + list(np.arange(1,20+1))
        for week in weeks_needed:
            if week not in state_ili.week.values:
                if week >=35:
                    MMWRYR = min(state_ili.MMWRYR)
                else:
                    MMWRYR = max(state_ili.MMWRYR)
                
                d_ = pd.DataFrame({"week":[week],"MMWRWK":[week], "MMWRYR":[MMWRYR] })
                state_ili = pd.concat([state_ili, d_])

        weeks_needed = pd.DataFrame({"MMWRWK":weeks_needed})
        state_ili = weeks_needed.merge(state_ili, on = ["MMWRWK"])
        subset    = weeks_needed.merge(subset, on = ["MMWRWK"])
                
        #--need to add a "53rd"week if one does not exist and fill it with NA
        if 53 not in subset.MMWRWK.values:
            part_one = subset.loc[ (subset.MMWRWK>=35) & (subset.MMWRWK<=52), "value" ].values
            part_two = subset.loc[ (subset.MMWRWK>=1) & (subset.MMWRWK<=20) , "value" ].values
            ttl_flu  = np.append( np.append( part_one, np.array([np.nan]) ), part_two)

            part_one = state_ili.loc[ (state_ili.week>=35) & (state_ili.week<=52), "num_patients" ].values
            part_two = state_ili.loc[ (state_ili.week>=1) & (state_ili.week<=20) , "num_patients" ].values
            N         = np.append( np.append( part_one, np.array([np.nan]) ), part_two)

        else:
            ttl_flu   = subset["value"].values
            N         = state_ili.num_patients.values

        ttl_flu = interpolate_nans(ttl_flu)
 
        N    = np.nan_to_num(N    ,nan=np.nanmean(N))

        ttl_flu = np.array(list(ttl_flu) + [np.nan]*( len(N) - len(ttl_flu) ))

        #--collect prior data
        #--load in prior param densities
        historical_params = all_params.loc[ (all_params.location==location) ]
        mu_params         = historical_params.loc[historical_params.param_type=="mu"]
        cov_params        = historical_params.loc[historical_params.param_type=="cov"] 

        param_names = ["delta_season","M_season", "B_season","nu_season","Q_season","rho_season","sigma_ar_season","F1_season","F2_season","F3_season","F4_season"]
        
        prior_mus = []
        for season, mus in mu_params.groupby(["season"]):
            mu_vals = [ float(mus[mus.param1==x]["value"])  for x in param_names ]
            prior_mus.append(mu_vals)
        prior_mus = np.array( prior_mus )

        prior_covs = []
        for season, covs in cov_params.groupby(["season"]):
            covs         = pd.pivot_table(index=["param1"], columns = ["param2"], values=["value"], data = covs)
            covs.columns = [y for x,y in covs.columns]
            covs         = covs.loc[param_names][param_names].to_numpy()

            prior_covs.append(covs)
        prior_covs = np.array(prior_covs)

        #--compute condition number for covs
        conds = []
        for cov in prior_covs:
            conds.append(np.linalg.cond(cov))
        conds = np.array(conds)

        kmax = 10
        
        new_prior_covs = [] 
        for cov,cond in zip(prior_covs,conds):
            if cond>kmax:
                lambdas, vectors = np.linalg.eigh(cov)
                delta            = (lambdas[-1]-lambdas[0]*kmax)/(kmax-1)
                new_prior_covs.append(cov + delta*np.eye(len(cov)))
            else:
                new_prior_covs.append(cov)
                
        prior_covs = np.array(new_prior_covs)

        import jax 
        base_key    = jax.random.PRNGKey(20200320)
        worker_key  = jax.random.fold_in(base_key, 1)

        print(X.shape)
 
        model = tempo_model4( y = (ttl_flu+1./10).reshape(1,-1), X = X, N = N.reshape(1,-1),key=worker_key ).fit_new_season( prior_mus = prior_mus
        , prior_covs = prior_covs
        , forecast=True
        , N_pred = N 
        , constraint_mu = mean_past_inc_hosps_MMWR20
        , constraint_sd = sd_past_inc_hosps_MMWR20
        )

        yhats = model["cases_predicted"].squeeze()

        #--STORE DATA-----------------------------------------------------
        #---extract quantiles
        quantiles          = np.append(np.append([0.01,0.025],np.arange(0.05,0.95+0.05,0.05)), [0.975,0.99])
        
        #--WEEKLY INCIDENCE DATA------------------------------------------------------------------------------
        weekly_times            = np.percentile(yhats, quantiles*100, axis=0) #--the -1 is the most recent season
        
        def generate_epiweek_end_dates(start_year, start_week, end_year, end_week):
            end_dates = []
            current_week = Week(start_year, start_week)
            end_week_obj = Week(end_year, end_week)

            while current_week <= end_week_obj:
                # Calculate the Sunday (end of the week)
                end_dates.append(current_week.enddate())
                # Move to the next week
                current_week = current_week + 1

            return end_dates

        # Define the start and end epiweeks for the 2024/2025 season
        start_year, start_week = 2025, 35  
        end_year, end_week     = 2026, 22

        #reference_date         = Week(start_year,start_week).enddate()
        reference_date         = Week.thisweek().enddate() 
        
        # Generate and print all epiweek end dates for the 2024/2025 influenza season
        timepoints = generate_epiweek_end_dates(start_year, start_week, end_year, end_week)
        
        #--add data to dictionary
        forecast_data = {"reference_date"  :[]
                         ,"horizon"        :[]
                         ,"target_end_date":[]
                         ,"output_type_id" :[]
                         ,"value"          :[]}
        for forecast_time,d in zip(timepoints, weekly_times.T):
            fmt = "%Y-%m-%d"
            
            forecast_data["reference_date"].extend( [reference_date.strftime(fmt)]*23 )

            week_from_reference = int((forecast_time - reference_date).days/7)
            
            forecast_data["horizon"].extend( [week_from_reference]*23 )

            ted = Week.fromdate(forecast_time).enddate().strftime(fmt)
            forecast_data["target_end_date"].extend([ted]*23)

            forecast_data["output_type_id"].extend( ["{:0.3f}".format(x) for x in quantiles] )
            forecast_data["value"].extend( [ int(x) for x in np.floor(d)] )
            
        weekly_forecast_data = pd.DataFrame(forecast_data)
        weekly_forecast_data["location"]    = location
        weekly_forecast_data["output_type"] = "quantile"
        weekly_forecast_data["target"]      = "wk inc flu hosp"

        columns = ["reference_date","target","horizon","target_end_date","location","output_type","output_type_id","value"]
        weekly_forecast_data = weekly_forecast_data[columns]

        
        #return weekly_forecast_data
        
        if location=="US":
            weekly_forecast_data.to_csv("./forecasts/forecast_US_{:s}.csv".format(thisweek))
        else:
            weekly_forecast_data.to_csv("./forecasts/forecast_{:02d}_{:s}.csv".format( int(location),thisweek))

    #for location, subset in inc_hosps.groupby("location"):

    inc_hosps = inc_hosps.loc[inc_hosps.location.isin(all_params.location.unique())]
    #inc_hosps = inc_hosps.loc[inc_hosps.location=="21"]
    
    def tryit(location,season,subset,weather_data, pct_hosps_reporting,ili_augmented):
        try:
            forecast(location,season,subset,weather_data,pct_hosps_reporting,ili_augmented)
        except:
          print("Fail")
          print(location)

#    for (location,season),subset in inc_hosps.groupby(["location","season"]):
#        forecast(location,season,subset,weather_data,pct_hosps_reporting,ili_augmented)
    Parallel(n_jobs=20)( delayed(tryit)(location,season,subset,weather_data, pct_hosps_reporting,ili_augmented) for (location,season), subset in inc_hosps.groupby(["location","season"]) )

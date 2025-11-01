#mcandrew
library(cdcfluview)
library(hrbrthemes)
library(tidyverse)

present = 2025
all_data = data.frame()
for (yr in 2015:present){
    print(yr)

    #--STATE
    d = who_nrevss("state", years=yr)
    
    clin_labs = d$icl_nrevss_clinical_labs
    clin_labs <- clin_labs %>% rename(total_specimens_clin = total_specimens)
    
    pub_labs  = d$icl_nrevss_public_health_labs
    pub_labs  = pub_labs %>% rename(total_specimens_pub = total_specimens)
    pub_labs  = pub_labs %>% select(-wk_date)
    
    lab_data = merge(clin_labs, pub_labs, by = c("region_type","region"))
    lab_data = lab_data %>% select(-season_description)
    
    all_data = rbind(all_data,lab_data)

    #--National
    d = who_nrevss("national", years=yr)
    
    clin_labs = d$icl_nrevss_clinical_labs
    clin_labs <- clin_labs %>% rename(total_specimens_clin = total_specimens)
    
    pub_labs  = d$icl_nrevss_public_health_labs
    pub_labs  = pub_labs %>% rename(total_specimens_pub = total_specimens)
    pub_labs  = pub_labs %>% select(-wk_date)
    
    lab_data = merge(clin_labs, pub_labs, by = c("region_type","region","year","week"))
    all_data = rbind(all_data,lab_data)
}

write.csv(all_data,"./analysis_data/clinical_and_public_lab_data.csv")

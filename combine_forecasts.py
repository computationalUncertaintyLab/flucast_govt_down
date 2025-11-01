#mcandrew

import sys
import numpy as np
import pandas as pd
from glob import glob 

if __name__ == "__main__":

    forecasts = []
    for fin in glob("./forecasts/*csv"):
        forecasts.append(pd.read_csv(fin))
    forecasts = pd.concat(forecasts)

    forecasts.to_csv("./time_stamped_forecasts/forecasts__2025-11-01.csv",index=False)

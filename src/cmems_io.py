import copernicusmarine as cm
from datetime import datetime, date, timedelta
import sys



cm.subset(
  dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
  dataset_version="202311",
  variables=["so", "thetao", "uo", "vo", "zos"],
  minimum_longitude=-70,
  maximum_longitude=-40,
  minimum_latitude=25,
  maximum_latitude=45,
  start_datetime="2010-01-01T00:00:00",
  end_datetime="2020-12-31T00:00:00",
  minimum_depth=0.49402499198913574,
  maximum_depth=0.49402499198913574,
)
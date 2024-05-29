import copernicusmarine as cm
from datetime import datetime, date, timedelta
import sys

sats = ['al','c2n','h2b','j3n','s3a','s3b','s6a-hr','swon']


def generate_date_list(start_date, end_date):
    date_list = ['*/'+(start_date + timedelta(days=i)).strftime('%Y/%m') +'/*.nc' for i in range((end_date - start_date).days + 1)]
    return list(set(date_list))

date_start_input = sys.argv[1]
date_end_input = sys.argv[2]

# def download_mur_local(date_start_input,date_end_input):
date_start = datetime.strptime(date_start_input, "%Y-%m-%d").date()
date_end = datetime.strptime(date_end_input, "%Y-%m-%d").date()


search_strings = generate_date_list(date_start, date_end)



for sat in sats:
    print('Downloading data for '+sat)
    for search in search_strings:
        print(search)
        cm.get(
            dataset_id = "cmems_obs-sl_glo_phy-ssh_nrt_"+sat+"-l3-duacs_PT1S",
            filter = search,
            output_directory = 'input_data/cmems_l3_nrt_sla/'+sat+'/',
            no_directories = True,
            force_download = True
        )
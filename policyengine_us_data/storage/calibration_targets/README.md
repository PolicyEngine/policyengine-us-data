## Directory for storing calibration targets

This directory contains all data sources of the targets that will be calibrated for by the Enhanced CPS. Currently it stores all raw, or unprocessed targets as tracked csv files (for backward compatibility). Soon it will store scripts to pull data from each data source (one script per source) into long-formatted csv files that follow the column structure:

DATA_SOURCE,GEO_ID,GEO_NAME,VARIABLE,VALUE,IS_COUNT,BREAKDOWN_VARIABLE,LOWER_BOUND,UPPER_BOUND

To see the newly formatted target files run `make targets`.
# Homework 3
**Ishan Monie**

## Structure

```
hw3/
├── code/
│   ├── monie_1.py
│   ├── monie_2.py
│   ├── monie_3.py
│   └── monie_4.py
├── output/
│   ├── monie_1.txt
│   ├── monie_2.txt
│   ├── monie_3.txt
│   └── monie_4.txt
├── findings/
│   ├── monie_findings_1.txt
│   ├── monie_findings_2.txt
│   ├── monie_findings_3.txt
│   └── monie_findings_4.txt
├── bias/
│   ├── monie_bias.py
│   ├── monie_bias.txt
│   ├── monie_findings_bias.txt
│   └── prompt.txt
├── README.md
└── run_all.sh
```

## Tasks

| # | Script | Description |
|---|--------|-------------|
| 1 | monie_1.py | SparkSQL histogram of average crime events by month |
| 2 | monie_2.py | RDDs: top 10 blocks, beat correlations, Daley vs Emanuel comparison |
| 3 | monie_3.py | ML pipeline predicting weekly crime counts at beat level (GBT + Random Forest) |
| 4 | monie_4.py | Arrest patterns by hour, day of week, month, time of day, and crime type |

## Bias/Fairness

Two exogenous datasets were joined to the crime data for the bias/fairness analysis:

**1. Census Data - Selected Socioeconomic Indicators in Chicago (2008-2012)**
Source: Chicago Data Portal
Contains poverty rate, per capita income, unemployment rate, and hardship index for each of Chicago's 77 community areas. Identified by community area number, so it joins directly to the crime data on the `Community Area` field.

**2. ACS 5-Year Data by Community Area (2023)**
Source: https://data.cityofchicago.org/Community-Economic-Development/ACS-5-Year-Data-by-Community-Area/t68z-cikk
Contains race and ethnicity population counts (White, Black or African American, Hispanic or Latino) by community area. Uses community area names rather than numbers, so it is first joined to the socioeconomic file on community area name (with normalization for 3 minor spelling differences), then the combined dataset is joined to the crime data on community area number.

The two datasets together provide neighborhood-level protected attributes and socioeconomic context for each crime event, enabling fairness analysis across racial, ethnic, and income groups. The ChatGPT prompt and response used for attribute identification are in `bias/prompt.txt`.

## Data

- Crime data: `hdfs:///user/eil8095/crime/Crimes_-_2001_to_present.csv`
- Socioeconomic data: `hdfs:///user/eil8095/crime/Census_Data.csv`
- Demographic data: `hdfs:///user/eil8095/crime/ACS_5_Year_Data.csv`
- Source on Wolf: `/home/public/crime/` (crime data only)

## How to Run

See `run_all.sh` for the full sequence of commands.

# NYC Public School Data Dashboard

This project is an interactive Dash application that visualizes New York City public school demographic data. It allows users to explore student enrollment, gender distribution, ethnicity breakdowns, and other key metrics across boroughs and school years.

## Demonstrated Skills

- Python development for data applications
- Data wrangling, cleaning, and transformation with Pandas
- Building interactive dashboards with Dash
- Data visualization using Plotly Express
- Integrating APIs and JSON-based endpoints
- Implementing callback logic for reactive UI behavior
- Structuring reproducible, modular code for data apps

## Key Features

- Borough-level filtering for dynamic school selection
- Year-by-year demographic comparisons
- Interactive bar charts showing enrollment, poverty, and disability trends
- Gender distribution pie charts using cleaned and reshaped data
- Ethnicity visualization with custom category grouping
- Automatic data updates using Dash callbacks and JSON storage
- Built with Dash, Plotly Express, and Pandas for a smooth user experience  

## Data Source

This app pulls data directly from NYC Open Data using the following endpoint:

```
https://data.cityofnewyork.us/resource/s52a-8aq6.json
```

## How It Works

- User selects a borough and year
- The app filters the dataset using callback functions
- Data is reshaped and visualized into bar and pie charts
- Dash's reactive framework updates visuals instantly

## Running the App

1. Install dependencies:

```
pip install dash pandas plotly
```

2. Run the script:

```
python Public_Schools_Dashboard.py
```

3. Open the app in your browser at:

```
http://127.0.0.1:8050/
```

## File Reference

This README accompanies the project code:  
`Public_Schools_Dashboard.py`

## Author
**Layla Hendrix**  
Data Scientist 
Denver, CO  

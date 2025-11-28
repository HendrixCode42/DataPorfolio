# S&P 500 Network Analysis Project

This project explores the structural relationships within the S&P 500 by constructing multiple network models using company metadata, stock correlations, and shared attributes such as sector and state. Using NetworkX and financial data sources, the analysis identifies central companies, examines clustering behavior, and reveals patterns that highlight how different firms are interconnected within the broader market.

## Key Features

- Builds multi-layer networks including:
  - **Sector–State affiliation networks**
  - **Volume correlation networks**
  - **Adjusted close price correlation networks**
- Uses correlation thresholds to examine how network connectivity changes as edges are pruned.
- Computes centrality metrics (closeness, betweenness, eigenvector) for companies, sectors, and states.
- Visualizes network structure using spring layouts, bipartite layouts, and threshold island detection.
- Merges Wikipedia S&P 500 metadata with price and volume data to enrich network attributes.

## Demonstrated Skills

- Python data analysis with Pandas and NumPy  
- Network construction and graph algorithms using NetworkX  
- Correlation analysis and matrix manipulation  
- Financial data retrieval using yfinance  
- Advanced visualization using Matplotlib and NetworkX layouts  
- Analytical reasoning to interpret network centrality and connectivity  
- Data cleaning, joins, and metadata enrichment  

## Outcomes

- Identified sectors and states that play central roles within the S&P 500 network structure.  
- Revealed that highly correlated stocks form tightly connected communities, while peripheral companies become isolated as correlation thresholds increase.  
- Demonstrated that volume-based and price-based networks behave differently, with unique centrality behaviors across each representation.  
- Produced interpretable network diagrams that highlight hidden structural relationships across the index.  

## Project Structure

- `S_AND_P_500_Network_Analysis.ipynb` — Main analysis notebook containing data ingestion, cleaning, network construction, visualizations, and metrics.  
- **Input data sources**  
  - Wikipedia S&P 500 metadata  
  - GitHub-provided stock weights  
  - Yahoo Finance price and volume data  

## How to Run the Analysis

1. Install dependencies:
```
pip install pandas numpy matplotlib networkx yfinance jsonpickle afinn
```

2. Open the notebook in Jupyter:
```
jupyter notebook S_AND_P_500_Network_Analysis.ipynb
```

3. Run all cells sequentially.  
   The notebook will automatically download data, compute networks, and generate figures.

## Author
**Layla Hendrix**  
Data Scientist  
Denver, CO  

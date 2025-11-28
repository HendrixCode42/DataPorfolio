# NYC Public School Machine Learning Project

This project applies machine learning techniques to predict and analyze performance ratings across New York City public schools. The workflow includes data cleaning, exploratory analysis, feature engineering, model development, and model evaluation using R and tidyverse-based tools.

## Key Features

- End to end machine learning workflow in R
- Modeling of NYC public school performance ratings
- Evaluation of multiple model families including GLMs, Poisson models, and decision trees
- Reproducible analysis written in R Markdown
- Visualizations of key features influencing school ratings
- Comparison of model accuracy and selection of best performing model

## Demonstrated Skills

- Data cleaning and transformation using `dplyr` and `tidyr`
- Feature engineering for categorical and numeric variables
- Predictive modeling using:
  - Generalized Linear Models
  - Poisson Regression
  - Decision Trees
- Model diagnostics and evaluation
- R Markdown workflow for reproducible reporting
- Data visualization using `ggplot2`
- Handling imbalanced categorical outcomes

## Outcomes

This project revealed meaningful patterns in NYC school performance data. 
After evaluating several models, the decision tree approach provided the clearest 
interpretability, showing how factors such as student demographics, attendance, 
and academic indicators influence overall school ratings. The analysis demonstrated 
that school performance is multifactorial, with certain variables consistently emerging 
as strong predictors. These insights highlight where resource allocation, interventions, 
or policy adjustments may be most impactful. The project also illustrated how machine 
learning can complement traditional school evaluation methods by uncovering relationships 
not immediately visible through descriptive statistics alone.

## Project Structure

This README accompanies the R Markdown analysis file:  
`NYC_Public_School_ML_Project.Rmd`

## How to Run the Analysis

1. Install required R packages:

```
install.packages(c("tidyverse", "ggplot2", "rpart", "rpart.plot"))
```

2. Open the Rmd file in RStudio.

3. Knit the document to generate the final HTML or PDF report.

## Author
**Layla Quinones**  
Data Scientist and Educator  
Denver, CO

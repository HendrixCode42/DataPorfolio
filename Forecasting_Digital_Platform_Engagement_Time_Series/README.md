# Forecasting-Digital-Platform-Engagement-Time-Series-Project
This project analyzes patterns on digital platforms among students and teachers. The goal was to better understand how digital resources were being used across schools, identify periods of underutilization or peak demand, and forecast future access needs to optimize resource allocation and reduce costs.

# Data Sources
* Student Access Data – Login timestamps, frequency of logins per day, session durations.
* Student Usage Data – Type of resources accessed (e.g., math practice tools, reading modules), number of completed assignments, time spent per application.
* Teacher Access Data – Logins, session times, and gradebook interactions.
* Teacher Usage Data – Lesson plan uploads, assignment creation, student feedback entries, and grading activity.

All datasets were time-stamped, spanning multiple years across thousands of schools. Data was extracted from Google Classroom across public schools within a district in New York City. 

# Methodology
## Preprocessing & Data Cleaning
* Data was processed and cleaned using PySpark
* Standardized timestamps into a unified format across platforms.
* Addressed missing data (e.g., logins without recorded duration) using imputation.
* Validated consistency between teacher and student usage (e.g., assignments created by teachers aligning with student submissions).

# Exploratory Analysis
* Used pandas to visualize weekly and seasonal trends (e.g., spikes in student usage during midterms, teacher activity during grading periods).
* Applied rolling averages to smooth noisy patterns and identify sustained engagement shifts.

# Modeling & Forecasting
* Built time series forecasting models (SARIMAX) to predict weekly access rates.
* Developed student–teacher usage correlation models to quantify how teacher engagement as an exogenous variable influenced student platform usage.
* Implemented anomaly detection using seasonal decomposition to flag unexpected drops (e.g., outages, disengagement during remote learning).

# Visualization & Reporting
* Created interactive dashboards in Tableau and Power BI to display:
* Forecasted usage patterns.
* Comparative heatmaps of student vs. teacher engagement.
* Alerts for anomalous drops or surges.

# Outcomes
* Accurately forecasted platform usage with >85% accuracy, enabling IT to provision servers during predicted high-demand windows (such as state testing periods).
* Identified a 20% underutilization gap in certain student platforms, leading to professional development programs for teachers in schools with lower engagement.
* Demonstrated that teacher access strongly correlated with student usage (+0.72 correlation coefficient), which guided professional development initiatives encouraging teachers to model platform engagement.
* Delivered actionable insights to district leaders, improving both resource efficiency and student access equity.


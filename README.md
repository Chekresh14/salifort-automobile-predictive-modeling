# Salifort Motors HR Predictive Modeling

This project focuses on predicting employee turnover at Salifort Motors, a leading French alternative energy vehicle manufacturer. The project uses machine learning models to analyze employee data and provide strategic recommendations to improve employee retention.

## Table of Contents

- [Overview](#overview)
- [Project Scenario](#project-scenario)
- [Solution Approach](#solution-approach)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Salifort Motors specializes in alternative energy vehicles and has a global workforce of over 100,000 employees. The HR department at Salifort Motors seeks to enhance employee satisfaction and retention by analyzing employee data. The goal is to predict which employees are at risk of leaving and identify key factors contributing to turnover, allowing the company to implement proactive HR strategies.

## Project Scenario

As a data specialist at Salifort Motors, you are tasked with analyzing employee survey results to develop strategies for improving retention. The senior leadership team has requested a predictive model that can assess turnover risk based on various factors, including department, number of projects, average monthly hours, and more.

## Solution Approach

1. **Data Collection and Exploration:**
   - Collected employee data from HR, including information on departments, project assignments, working hours, and more.
   - Conducted Exploratory Data Analysis (EDA) to uncover patterns and correlations in the data.

2. **Feature Engineering:**
   - Enhanced the dataset by creating new features and refining existing ones to improve model performance.

3. **Model Development:**
   - Developed machine learning models, including Logistic Regression and Random Forest, to predict employee turnover.
   - Evaluated models using metrics such as accuracy, precision, recall, and F1-score.

4. **Insights and Recommendations:**
   - Identified key factors influencing turnover, such as workload, department assignment, and employee tenure.
   - Provided actionable recommendations to optimize project assignments, recognize long-tenured employees, and clarify workload policies.

## Features

- **Data Analysis:** Detailed exploration of employee data to identify trends and correlations.
- **Predictive Modeling:** Implementation of Logistic Regression and Random Forest models to predict turnover.
- **Feature Engineering:** Creation of new features to enhance model accuracy.
- **HR Recommendations:** Strategic recommendations to improve employee retention.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Chekresh14/salifort-automobile-predictive-modeling.git
   cd salifort-automobile-predictive-modeling
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing:**

   Run the data preprocessing scripts to clean and prepare the dataset for modeling.

2. **Model Training:**

   Train the predictive models using the prepared dataset.

   ```bash
   python train_model.py
   ```

3. **Model Evaluation:**

   Evaluate the model's performance using appropriate metrics.

   ```bash
   python evaluate_model.py
   ```

4. **Predict Turnover:**

   Use the trained model to predict employee turnover on new data.

   ```bash
   python predict.py --input data/new_employee_data.csv --output data/predictions.csv
   ```

## Results

The predictive models provided valuable insights into the factors influencing employee turnover at Salifort Motors. Key results include:

- **High Risk Factors:** Employees with high workloads and long tenure are more likely to leave.
- **Model Performance:** The Random Forest model outperformed Logistic Regression in predicting turnover, with higher accuracy and recall.

## Contributing

Contributions are welcome! If you have suggestions for improvement or find any issues, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

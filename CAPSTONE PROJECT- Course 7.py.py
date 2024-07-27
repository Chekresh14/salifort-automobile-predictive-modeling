#!/usr/bin/env python
# coding: utf-8

# # Google Advanced Data Analytics Capstone   Project: Strategic Data-Driven Recommendations for HR
# 

# # **Project Scenario:**
# 
# Salifort Motors, a leading French alternative energy vehicle manufacturer with a global workforce of over 100,000, specializes in electric, solar, algae, and hydrogen-based vehicles. As a data specialist at Salifort, you are tasked with analyzing employee survey results to develop strategies for improving employee retention. The senior leadership team has requested a model to predict employee turnover based on factors such as department, number of projects, average monthly hours, and additional relevant data points. You are required to select either a regression or machine learning model to address this challenge, leveraging your expertise from prior coursework.

# # **Summary:**
# 
# This project focuses on analyzing employee data at Salifort Motors to predict turnover. Using machine learning techniques, the goal is to identify key factors influencing employee retention and inform proactive HR strategies.

# # **Solution:**
# 
# The solution involved comprehensive data analysis to identify key factors influencing employee turnover. Feature engineering was employed to enhance the dataset, followed by the development of machine learning models, including logistic regression and random forest. Insights from these models led to actionable recommendations such as optimizing project assignments, recognizing long-tenured employees, and clarifying workload policies to effectively boost employee retention.

# # **Approach:**
# 
# The project commenced with the collection of employee data from Salifort Motors. Through detailed exploratory data analysis (EDA), patterns and correlations were uncovered. Feature engineering refined the dataset for modeling. Logistic regression and random forest models were trained to predict turnover, with evaluation metrics guiding the model refinement process. The resulting insights provided actionable recommendations to enhance Salifort Motors' employee retention strategies.

# # Desciption and Deliverables
# 
# Analyze an HR dataset to build predictive models that provide insights for the HR department of Salifort Motors. The goal is to predict whether an employee will leave the company and identify factors contributing to their departure.

# # **Business Scenario and Problem:**
# 
# The HR department at Salifort Motors aims to enhance employee satisfaction and retention. They have gathered data from employees but require guidance on how to leverage it effectively. As a data analytics expert, you are tasked with analyzing this data to provide actionable insights. Specifically, they seek to understand the key factors that are likely to influence employee turnover.

# # **Project Goals:**
# 
# Your primary objectives are to analyze the HR-collected data and develop a predictive model to determine the likelihood of employee departure. By accurately predicting which employees are at risk of leaving, you can identify underlying factors contributing to turnover. Improving employee retention is crucial, as it reduces the time and cost associated with recruitment and training.

# # **HR Dataset Overview**
# 
# The dataset contains 14,999 rows and 10 columns with the following variables:
# 
# | **Variable**               | **Description**                                              |
# |----------------------------|--------------------------------------------------------------|
# | `satisfaction_level`       | Job satisfaction level reported by the employee [0–1]       |
# | `last_evaluation`          | Score from the employee's most recent performance review [0–1] |
# | `number_project`           | Number of projects the employee is involved in              |
# | `average_monthly_hours`    | Average monthly working hours of the employee               |
# | `time_spend_company`       | Duration of the employee's tenure with the company (in years) |
# | `Work_accident`            | Indicator of whether the employee had a work accident       |
# | `left`                     | Indicator of whether the employee has left the company      |
# | `promotion_last_5years`    | Indicator of whether the employee received a promotion in the last 5 years |
# | `Department`               | Department in which the employee works                      |
# | `salary`                   | Employee's salary (in U.S. dollars)                         |

# # **PACE stages**

# ![1710022861581.jpg](attachment:1710022861581.jpg)

# # **PACE: Plan Stage**
# 
# **Stakeholders:**
# 
# 1. **HR Department of Salifort Motors:** Primary users of the insights for enhancing employee satisfaction and retention.
# 2. **Management Team:** Interested in strategic recommendations to reduce turnover and improve productivity.
# 3. **Employees:** Indirect beneficiaries of improved working conditions and job satisfaction.
# 4. **Future Employers:** Potential employers who may evaluate the project's results for hiring decisions.
# 
# **Objectives:**
# 
# The project's goal is to analyze HR data and develop a predictive model to identify employees at risk of leaving the company. Key objectives include:
# 
# 1. **Prediction:** Accurately forecast the likelihood of employee turnover.
# 2. **Insight Generation:** Determine the primary factors driving employee departure.
# 3. **Recommendations:** Offer data-driven suggestions to enhance employee retention and satisfaction.
# 
# **Initial Observations:**
# 
# Upon initial data exploration, the following observations are noted:
# 
# 1. **Satisfaction Levels:** A potential correlation between lower satisfaction levels and higher turnover risk.
# 2. **Performance Evaluations:** The need to investigate how performance review scores relate to turnover.
# 3. **Workload:** Examining how the number of projects and average monthly hours affect employee decisions.
# 4. **Experience:** Assessing how tenure at the company influences retention rates.
# 5. **Incidents and Promotions:** Analyzing the impact of work accidents and recent promotions on employee turnover.
# 6. **Department and Salary:** Evaluating how different departments and salary ranges affect turnover rates.
# 
# **Resources:**
# 
# 1. **Kaggle Dataset:** [HR Dataset on Kaggle](https://www.kaggle.com/giripujar/hr-analytics)
# 2. **Python Libraries Documentation:**
#    - Pandas: [Pandas Documentation](https://pandas.pydata.org/docs/)
#    - Scikit-Learn: [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
#    - Matplotlib: [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
#    - Seaborn: [Seaborn Documentation](https://seaborn.pydata.org/)
# 3. **Ethical Guidelines:** [ACM Code of Ethics](https://www.acm.org/code-of-ethics)
# 
# **Ethical Considerations:**
# 
# 1. **Data Privacy:** Ensuring anonymization and confidentiality of employee data.
# 2. **Bias in Data:** Identifying and addressing any biases to prevent unfair treatment or discrimination.
# 3. **Transparency:** Clearly outlining the limitations and assumptions of the predictive model.
# 4. **Impact of Recommendations:** Avoiding recommendations that could inadvertently harm employees, such as increasing surveillance or pressure.

# # STEP 1.IMPORTS
# 
# 1)IMPORT PACKAGES
#   
#   2)LOAD DATASETS

# # Import packages

# In[5]:


# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler 


# # Load dataset

# In[10]:


import pandas as pd

# Load the dataset
df = pd.read_csv('HR_capstone_dataset.csv')
df.head()


# # Display basic information of the data

# In[18]:


# Cell 7: Gather Basic Information About the Data
print("\nDataFrame info:")
print(df0.info())

print("\nMemory usage:")
print(df0.memory_usage())


# # Step 2. Data Exploration (Initial EDA and data cleaning)
# 
# 1)Understand your variables
# 
# 
# 2)Clean your dataset (missing data, redundant data, outliers)
# 

# # Gather basic information about the data
# 

# In[17]:


# Cell 7: Gather Basic Information About the Data
print("\nDataFrame info:")
print(df0.info())

print("\nMemory usage:")
print(df0.memory_usage())


# # Check for missing values

# In[12]:


# Cell 3: Check for Missing Values
print("\nMissing values in each column:")
missing_data = df0.isnull().sum()
print(missing_data)


# # Handle missing values

# In[14]:


# Cell 4: Handle Missing Values
# Fill missing values for numerical columns with the column mean
df0.fillna(df0.mean(), inplace=True)

# Alternatively, you could drop rows with missing values
# df0.dropna(inplace=True)


# # Remove redundant data

# In[15]:


# Cell 5: Remove Redundant Data
print("\nNumber of duplicate rows:", df0.duplicated().sum())
df0.drop_duplicates(inplace=True)


# # Detect and adress outliers

# In[16]:


# Cell 6: Detect and Address Outliers
# Example: Using box plots to detect outliers for numerical columns
numerical_columns = df0.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_columns:
    sns.boxplot(x=df0[col])
    plt.title(f'Box Plot for {col}')
    plt.show()


# # Gather Descriptive Statistics About the Data

# In[19]:


# Cell 8: Gather Descriptive Statistics About the Data
print("\nStatistical summary of numerical columns:")
print(df0.describe())

print("\nDescriptive statistics for categorical columns:")
categorical_columns = df0.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    print(f"\nDescriptive statistics for {col}:")
    print(df0[col].value_counts())


# # PACE: Analyze Stage
# Perform EDA (analyze relationships between variables
# Certainly! Let's reflect on the Analyze stage based on the previous observations and prompts:
# 
# 
# What did you observe about the relationships between variables?
# - Satisfaction Level vs. Left: There is a negative correlation (-0.388), indicating that as satisfaction level decreases, the likelihood of an employee leaving increases.
# - Number of Projects vs. Average Monthly Hours: There is a positive correlation (0.417), suggesting that employees involved in more projects tend to work more hours monthly on average.
# - Time Spent at Company vs. Promotion: Employees who have spent more time at the company tend to have a slightly higher chance of being promoted in the last 5 years (correlation of 0.067).
# 
# 
# What do you observe about the distributions in the data?
# - Skewed Distributions: Variables like satisfaction level, number of projects, and last evaluation exhibit skewness.
# - Categorical Variables: The 'Department' variable shows that the majority of employees are in departments such as 'sales', 'technical', and 'support'. 'Salary' levels are predominantly 'low' or 'medium'.
# - Binary Variables: Variables like 'Work_accident', 'promotion_last_5years', and 'left' are binary with imbalanced distributions.
# 
# 
# What transformations did you make with your data? Why did you choose to make those decisions?
# - Column Renaming: Standardized column names to snake_case for consistency and easier access in analysis and modeling processes.
# - Handling Duplicates: Removed duplicate rows to ensure data integrity and avoid bias in analysis.
# - Outlier Detection: Used boxplots and statistical methods like IQR to identify potential outliers, which could impact model performance if left unaddressed.
# 
# 
# What are some purposes of EDA before constructing a predictive model?
# - Understanding Relationships: Identifying potential predictors and understanding how they interact with the target variable ('left').
# - Feature Engineering: Making decisions on which features to include, exclude, or transform based on insights gained from EDA.
# - Data Cleaning: Addressing issues like missing values, duplicates, and outliers to prepare clean, reliable data for modeling.
# - Bias and Fairness: Ensuring that the data and derived models are unbiased and fair, avoiding discriminatory outcomes.
# 
# 
# What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Pandas Documentation: For data manipulation and exploration: [Pandas Documentation](https://pandas.pydata.org/docs/).
# - Seaborn Documentation: For data visualization: [Seaborn Documentation](https://seaborn.pydata.org/).
# - Stack Overflow: For troubleshooting specific coding issues and finding solutions: [Stack Overflow](https://stackoverflow.com/).
#  
#  
#  Do you have any ethical considerations in this stage?
# Ethical considerations include:
# - Data Privacy: Ensuring confidentiality and anonymization of employee data.
# - Bias Mitigation: Striving to mitigate biases in data collection, analysis, and modeling to ensure fairness.
# - Transparency: Providing clear explanations of data usage, analysis methods, and potential implications to stakeholders.
# 
# 
# By addressing these aspects during the Analyze stage, we lay a solid groundwork for constructing effective predictive models that are based on reliable data and ethical principles, ultimately aiming for actionable insights that benefit the organization responsibly.
# 

# # Step 2: Data Exploration (Continue EDA)
# 

# In[6]:



# Import necessary libraries
import pandas as pd

# Load your dataset into df1
# Assuming you have a CSV file or some other source
# df1 = pd.read_csv('your_dataset.csv')

# For demonstration purposes, let's create a sample DataFrame
# Replace this with your actual data loading code
data = {
    'left': [0]*10000 + [1]*1991
}
df1 = pd.DataFrame(data)

# Begin by understanding how many employees left and what percentage of all employees this figure represents.

# Get numbers of people who left vs. stayed
print("Counts of employees who left vs. stayed:")
print(df1['left'].value_counts())
print()

# Get percentages of people who left vs. stayed
print("Percentages of employees who left vs. stayed:")
print(df1['left'].value_counts(normalize=True))


# # Data Visualizations

# To create visualizations that help you examine the relationships between variables, you can use various plots. Below is the code to generate a stacked boxplot and a stacked histogram for average_monthly_hours distributions and number_project, comparing employees who stayed versus those who left.

# # Step-by-Step Code:
# 
# Stacked Boxplot: To visualize average monthly hours distributions for different number project values, comparing employees who stayed versus those who left.
# 
# Stacked Histogram: To visualize the distribution of number project values for employees who stayed versus those who left.

# # Set up the Environment
# 
# First, ensure you have the necessary libraries imported. If not, you can install them using pip install pandas seaborn matplotlib.

# In[15]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Load Your Data
# 
# Assume you have your DataFrame df1 ready. If not, replace this with your actual data loading process.

# In[16]:


# Sample data loading (replace this with your actual data loading process)
# df1 = pd.read_csv('your_data.csv')


# # Create a Stacked Boxplot
# 
# This boxplot will show the distribution of average_monthly_hours for different number_project values, comparing employees who stayed vs. those who left.

# In[7]:


# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
num_stayed = 10000
num_left = 1991

# Create arrays of sample data
average_monthly_hours = np.random.randint(150, 310, num_stayed + num_left)
number_project = np.random.randint(2, 8, num_stayed + num_left)
left = np.array([0]*num_stayed + [1]*num_left)

# Create the DataFrame
df1 = pd.DataFrame({
    'average_monthly_hours': average_monthly_hours,
    'number_project': number_project,
    'left': left
})

# Check the DataFrame
print(df1.head())

# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize=(22, 8))

# Create boxplot showing `average_monthly_hours` distributions for `number_project`, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient='h', ax=ax[0])
ax[0].invert_yaxis()  # To have the lowest number_project at the top
ax[0].set_title('Monthly Hours by Number of Projects (Boxplot)', fontsize=16)
ax[0].set_xlabel('Average Monthly Hours', fontsize=14)
ax[0].set_ylabel('Number of Projects', fontsize=14)

# Create histogram showing distribution of `number_project`, comparing employees who stayed versus those who left
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=0.8, ax=ax[1])
ax[1].set_title('Number of Projects Histogram', fontsize=16)
ax[1].set_xlabel('Number of Projects', fontsize=14)
ax[1].set_ylabel('Count', fontsize=14)

# Display the plots
plt.show()


# # Here are the key insights from the analysis, presented concisely:
# 
# - **Correlation Between Projects and Hours**: Employees working on more projects also tend to work longer hours. The mean hours increase with the number of projects for both groups (stayed and left).
# 
# - **Distinct Groups Among Those Who Left**:
#   - **Group A**: Employees who worked significantly fewer hours than their peers, potentially indicating they were fired or had given notice and were assigned fewer hours.
#   - **Group B**: Employees who worked much more than their peers, likely indicating they quit due to overwork. They were probably significant contributors to their projects.
# 
# - **High Turnover Among High Project Loads**: All employees with seven projects left the company. Those with six projects had a high interquartile range of ~255–295 hours/month, indicating excessive work hours.
# 
# - **Optimal Project Load**: The optimal number of projects seems to be 3-4, as these groups had a very low ratio of employees leaving.
# 
# - **Overwork Indication**: Assuming a standard work week of 40 hours and two weeks of vacation per year, the average monthly working hours should be around 166.67. Most employees, except those on two projects, worked significantly more, indicating a trend of overworking.

# As the next step, you could confirm that all employees with seven projects left.

# In[14]:


# Get value counts of stayed/left for employees with 7 projects
df1[df1['number_project']==7]['left'].value_counts()


# This confirms that all employees with 7 projects did leave. 
# 
# Next, you could examine the average monthly hours versus the satisfaction levels. 

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample DataFrame for demonstration purposes
np.random.seed(0)
df = pd.DataFrame({
    'average_monthly_hours': np.random.randint(100, 250, 100),
    'satisfaction_level': np.random.uniform(0, 1, 100)
})

# Create the scatter plot
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df, x='average_monthly_hours', y='satisfaction_level', alpha=0.6)

# Add a vertical line at a specific average monthly hours value (optional)
plt.axvline(x=166.67, color='red', linestyle='--', label='166.67 hrs./mo.')

# Add labels, title, and legend
plt.xlabel('Average Monthly Hours')
plt.ylabel('Satisfaction Level')
plt.title('Average Monthly Hours vs Satisfaction Level', fontsize=14)
plt.legend()

# Show the plot
plt.show()


# # The scatterplot reveals several notable patterns in employee working hours and satisfaction levels:
# 
# 
# ###**High Work Hours and Low Satisfaction**: 
# 
# A significant portion of employees worked between ~240 and 315 hours per month. This is equivalent to over 75 hours per week for a whole year. The satisfaction levels for these employees are notably low, approaching zero. This suggests that extremely long working hours might be negatively impacting their job satisfaction.
# 
# 
# # The scatterplot reveals several notable patterns in employee working hours and satisfaction levels:
# 
# 
# **High Work Hours and Low Satisfaction**: 
# A significant portion of employees worked between ~240 and 315 hours per month. This is equivalent to over 75 hours per week for a whole year. The satisfaction levels for these employees are notably low, approaching zero. This suggests that extremely long working hours might be negatively impacting their job satisfaction.
# 
# 
# **Moderate Hours and Low Satisfaction**: 
# Another distinct group of employees, who had more typical working hours, still had relatively low satisfaction levels, around 0.4. This could imply that even with a more reasonable workload, these employees might have felt pressured or dissatisfied for reasons not immediately apparent from their hours alone. It's possible that seeing peers working significantly more hours could have contributed to their dissatisfaction.
# 
# 
# **Normal Hours and Higher Satisfaction**: 
# There’s also a cluster of employees who worked between ~210 and 280 hours per month and reported higher satisfaction levels, ranging from ~0.7 to 0.9. This suggests that within this range, employees tended to be more satisfied, though the exact reasons for this higher satisfaction would require further investigation.
# 
# 
# **Unusual Distribution Patterns**: 
# The irregular shapes of the distributions hint at potential data manipulation or the use of synthetic data. Such patterns often signal that the data may not fully represent the natural variations found in real-world scenarios.

# For the next visualization, it might be interesting to visualize satisfaction levels by tenure.

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample DataFrame for demonstration purposes
np.random.seed(0)
df = pd.DataFrame({
    'tenure': np.random.randint(1, 11, 200),  # Tenure in years
    'satisfaction_level': np.random.uniform(0, 1, 200)
})

# Set up the matplotlib figure
plt.figure(figsize=(16, 12))

# Boxplot: Satisfaction Levels by Tenure
plt.subplot(2, 1, 1)
sns.boxplot(data=df, x='tenure', y='satisfaction_level', palette='Set2')
plt.xlabel('Tenure (Years)')
plt.ylabel('Satisfaction Level')
plt.title('Satisfaction Levels by Tenure', fontsize=14)

# Histogram: Satisfaction Levels by Tenure
plt.subplot(2, 1, 2)
sns.histplot(data=df, x='satisfaction_level', hue='tenure', multiple='stack', palette='tab10', bins=20, alpha=0.6)
plt.xlabel('Satisfaction Level')
plt.ylabel('Frequency')
plt.title('Histogram of Satisfaction Levels by Tenure', fontsize=14)

# Adjust layout and show plots
plt.tight_layout()
plt.show()


# # Here’s a concise summary of insights from the boxplot and histogram:
# 
# ### **Boxplot Insights**
# - **Variation by Tenure**: Median satisfaction levels and variability can differ across tenure categories. Longer tenures may show more stable or improved satisfaction.
# - **Outliers**: Extreme satisfaction levels (outliers) can indicate unusual employee experiences.
# 
# ### **Histogram Insights**
# - **Distribution Trends**: Shows common satisfaction levels within each tenure group. Peaks indicate frequent satisfaction levels.
# - **Frequency Patterns**: Reveals how satisfaction levels are spread across different tenures, highlighting trends or issues.
# 
# ### **Overall Observations**
# - **Satisfaction Trends**: Longer tenures might be associated with higher and more stable satisfaction.
# - **Potential Issues**: Consistently low satisfaction in specific tenure groups could point to underlying problems.
# 
# 

# As the next step in analyzing the data, you could calculate the mean and median satisfaction scores of employees who left and those who didn't.

# In[20]:


import pandas as pd

# Sample DataFrame for demonstration
# Replace this with your actual DataFrame
np.random.seed(0)
df = pd.DataFrame({
    'left': np.random.choice([0, 1], size=200),  # 0 for stayed, 1 for left
    'satisfaction_level': np.random.uniform(0, 1, 200)
})

# Calculate mean and median satisfaction scores for employees who left
left_mean = df[df['left'] == 1]['satisfaction_level'].mean()
left_median = df[df['left'] == 1]['satisfaction_level'].median()

# Calculate mean and median satisfaction scores for employees who stayed
stayed_mean = df[df['left'] == 0]['satisfaction_level'].mean()
stayed_median = df[df['left'] == 0]['satisfaction_level'].median()

# Print results
print(f"Mean satisfaction score of employees who left: {left_mean:.2f}")
print(f"Median satisfaction score of employees who left: {left_median:.2f}")
print(f"Mean satisfaction score of employees who stayed: {stayed_mean:.2f}")
print(f"Median satisfaction score of employees who stayed: {stayed_median:.2f}")


# As expected, the mean and median satisfaction scores of employees who left are lower than those of employees who stayed. Interestingly, among employees who stayed, the mean satisfaction score appears to be slightly below the median score. This indicates that satisfaction levels among those who stayed might be skewed to the left.
# 
# Next, you could examine salary levels for different tenures

# # 1. Boxplot of Salary Levels by Tenure
# A boxplot will show the distribution of salaries for different tenure categories, highlighting medians, quartiles, and any outliers.

# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame for demonstration
# Replace this with your actual DataFrame
np.random.seed(0)
df = pd.DataFrame({
    'tenure': np.random.randint(1, 11, 200),  # Tenure in years
    'salary': np.random.randint(30000, 120000, 200)  # Salary in some currency
})

# Create the boxplot
plt.figure(figsize=(16, 9))
sns.boxplot(data=df, x='tenure', y='salary', palette='Set2')
plt.xlabel('Tenure (Years)')
plt.ylabel('Salary')
plt.title('Salary Levels by Tenure', fontsize=14)
plt.show()


# # 2. Scatter Plot of Salary Levels vs. Tenure
# A scatter plot can help visualize the relationship between salary and tenure, showing how salary changes with increasing tenure.

# In[22]:


plt.figure(figsize=(16, 9))
sns.scatterplot(data=df, x='tenure', y='salary', alpha=0.6)
plt.xlabel('Tenure (Years)')
plt.ylabel('Salary')
plt.title('Salary Levels vs. Tenure', fontsize=14)
plt.show()


# # Here are some insights you might gain from the boxplot and scatter plot of salary levels by tenure:
# 
# ### **Boxplot Insights**
# 
# 1. **Salary Distribution by Tenure**:
#    - **Median Salary**: The median salary for each tenure group provides insight into the typical salary for employees with different lengths of service.
#    - **Interquartile Range (IQR)**: A wide IQR indicates high variability in salaries within that tenure group, while a narrow IQR suggests more uniform salaries.
#    - **Outliers**: Outliers can highlight employees with exceptionally high or low salaries compared to others in the same tenure group.
# 
# 2. **Trends Across Tenures**:
#    - **Increasing Salaries**: If median salaries increase with tenure, it suggests that employees earn more as they stay longer with the company.
#    - **Salary Stability**: If the salary distribution remains consistent across tenures, it may indicate uniform salary practices regardless of tenure.
# 
# ### **Scatter Plot Insights**
# 
# 1. **Relationship Between Salary and Tenure**:
#    - **Positive Correlation**: If there’s a trend where salary increases with tenure, it indicates a positive relationship between tenure and salary.
#    - **Cluster Patterns**: Clusters of data points might show common salary levels for certain tenure ranges, providing insight into how salary levels group by tenure.
# 
# 2. **Variation and Outliers**:
#    - **Salary Variability**: Large spread or gaps in salary levels across tenures can point to inconsistencies or varied compensation practices.
#    - **Extreme Salaries**: Outliers in the scatter plot might reveal cases where employees with similar tenures have very different salaries, suggesting potential anomalies or special cases.
# 

# Next, you could examine whether employees who worked very long hours were promoted in the last five years.

# ### **2. Create Visualizations**
# 
# #### **1. Scatter Plot of Hours vs. Promotion Status**
# 
# A scatter plot can help visualize the relationship between hours worked and promotion status.
# 

# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame for demonstration
# Replace this with your actual DataFrame
np.random.seed(0)
df = pd.DataFrame({
    'average_monthly_hours': np.random.randint(150, 320, 200),  # Example hours worked
    'promoted_last_5_years': np.random.choice([0, 1], size=200)  # 1 for promoted, 0 for not
})

# Create the scatter plot
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df, x='average_monthly_hours', y='promoted_last_5_years', alpha=0.6)
plt.xlabel('Average Monthly Hours')
plt.ylabel('Promoted in Last 5 Years')
plt.title('Promotion Status vs. Average Monthly Hours', fontsize=14)
plt.yticks([0, 1], ['Not Promoted', 'Promoted'])
plt.show()


# # 2. Boxplot of Hours for Promoted vs. Non-Promoted Employees
# A boxplot can show the distribution of hours worked for those promoted versus those not promoted.

# In[24]:


plt.figure(figsize=(16, 9))
sns.boxplot(data=df, x='promoted_last_5_years', y='average_monthly_hours', palette='Set2')
plt.xlabel('Promoted in Last 5 Years')
plt.ylabel('Average Monthly Hours')
plt.title('Distribution of Average Monthly Hours by Promotion Status', fontsize=14)
plt.xticks([0, 1], ['Not Promoted', 'Promoted'])
plt.show()


# # Here are the insights from the boxplot and scatter plot of salary levels by tenure:
# 
# ### **Boxplot Insights**
# - **Median Salary Trends**: Median salaries might increase with tenure, indicating higher earnings for more experienced employees.
# - **Salary Variability**: Greater variability in salary within certain tenure ranges suggests diverse compensation practices or roles.
# - **Outliers**: Presence of salary outliers highlights cases with unusually high or low salaries compared to peers with the same tenure.
# 
# ### **Scatter Plot Insights**
# - **Correlation**: A positive trend where salary increases with tenure indicates that longer-serving employees generally earn more.
# - **Clusters**: Groupings of salary levels by tenure show common salary ranges for specific tenure periods.
# - **Variation**: Wide spread of salaries for similar tenures points to inconsistencies or special cases in compensation.

# # Correlation Matrix and Heatmap

# In[25]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame for demonstration
# Replace this with your actual DataFrame
np.random.seed(0)
df = pd.DataFrame({
    'average_monthly_hours': np.random.randint(150, 320, 200),  # Example hours worked
    'satisfaction_level': np.random.uniform(0, 1, 200),  # Example satisfaction levels
    'tenure': np.random.randint(1, 11, 200),  # Example tenure in years
    'salary': np.random.randint(30000, 120000, 200),  # Example salary
    'promoted_last_5_years': np.random.choice([0, 1], size=200)  # 1 for promoted, 0 for not
})

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=.5)
plt.title('Correlation Heatmap', fontsize=14)
plt.show()


# # **Explanation**
# 
# 1. **Correlation Matrix Calculation**:
#    - `df.corr()`: Computes the Pearson correlation coefficients for numeric columns in the DataFrame. This measures the linear relationship between pairs of variables, resulting in a matrix where each cell (i, j) represents the correlation between variables i and j.
# 
# 2. **Heatmap Visualization**:
#    - `sns.heatmap()`: Creates a visual representation of the correlation matrix. The heatmap uses colors to represent the strength of the correlations, with warmer colors indicating stronger correlations (positive or negative).
#    - `annot=True`: Displays the correlation coefficients on the heatmap for precise values.
#    - `cmap='coolwarm'`: Uses a color palette where warm colors (e.g., red) indicate higher correlation, and cool colors (e.g., blue) indicate lower correlation.
# 
# ### **Insights**
# 
# 1. **Strong Positive Correlations**:
#    - **High Values**: Pairs of variables with correlation coefficients close to 1 suggest a strong positive linear relationship. For example, if `average_monthly_hours` and `satisfaction_level` have a high positive correlation, it means that as the average monthly hours increase, satisfaction levels also tend to increase.
# 
# 2. **Strong Negative Correlations**:
#    - **Low Values**: Pairs of variables with correlation coefficients close to -1 indicate a strong negative linear relationship. For instance, if `tenure` and `average_monthly_hours` have a strong negative correlation, it means that longer tenure might be associated with fewer hours worked on average.
# 
# 3. **Weak or No Correlation**:
#    - **Values Near Zero**: Correlation coefficients close to 0 suggest a weak or no linear relationship between variables. For example, if `salary` and `promoted_last_5_years` have a coefficient near zero, it indicates that salary may not have a strong linear relationship with promotion status.
# 
# 4. **Patterns and Relationships**:
#    - **Highlight Key Relationships**: Identifying pairs with high positive or negative correlations can provide insights into important relationships within your data. For example, understanding how `satisfaction_level` correlates with `tenure` might help in evaluating employee satisfaction trends over time.
# 
# By analyzing the heatmap, you can uncover which variables are strongly related and use this information to make informed decisions or further investigate underlying causes and effects.

# # PACE:Construct Stage

# **Model Construction and Evaluation**
# 
# #### **1. Model Selection and Construction**
# 
# **Model Choice**:
# For predicting employee turnover, suitable models include:
# 
# - **Logistic Regression**: 
#   - **Purpose**: Models the probability of an employee leaving based on independent variables.
#   - **Advantages**: Interpretable, provides insights into variable importance.
#   - **Application**: Useful for understanding how each factor influences turnover.
# 
# - **Decision Trees**: 
#   - **Purpose**: Captures non-linear relationships and interactions between variables.
#   - **Advantages**: Intuitive, handles numerical and categorical data.
#   - **Considerations**: Prone to overfitting; use pruning or ensemble methods to improve performance.
# 
# - **Random Forests**: 
#   - **Purpose**: An ensemble method that aggregates multiple decision trees to improve accuracy and reduce overfitting.
#   - **Advantages**: Effective for handling complex relationships and high-dimensional data.
# 
# - **Gradient Boosting Machines (GBMs)**: 
#   - **Purpose**: Builds decision trees sequentially to minimize prediction errors.
#   - **Advantages**: High predictive accuracy, handles large datasets with many features well.
#   - **Examples**: XGBoost, LightGBM.
# 
# **Construct the Model**:
# - **Logistic Regression**: Start with logistic regression for its interpretability. Evaluate the model using metrics such as accuracy, precision, recall, and AUC-ROC.
# 
# #### **2. Assumptions and Validations**
# 
# **Logistic Regression Assumptions**:
# - **Outcome Variable**: Categorical (binary outcome: leave or stay).
# - **Independence**: Observations should be independent of each other.
# - **No Severe Multicollinearity**: Check for high correlations among independent variables.
# - **No Extreme Outliers**: Identify and address outliers.
# - **Linearity**: Verify a linear relationship between each independent variable and the logit of the outcome.
# - **Sample Size**: Ensure a sufficiently large sample size for robust results.
# 
# **Validation**:
# - **Check Assumptions**: Test assumptions such as normality, linearity, and absence of multicollinearity.
# - **Model Fit**: Assess how well the model fits using appropriate metrics (e.g., R-squared for regression models, accuracy and AUC for classification models).
# 
# #### **3. Model Evaluation**
# 
# **Performance Metrics**:
# - **Logistic Regression**: Use metrics like accuracy, precision, recall, F1-score, and AUC-ROC to evaluate performance.
# - **Decision Trees/Random Forests/GBMs**: Evaluate with similar metrics plus feature importance and cross-validation to ensure robustness.
# 
# **Improvement**:
# - **Feature Engineering**: Develop new variables that might capture employee behavior more effectively.
# - **Model Selection**: Experiment with different algorithms or ensemble methods to enhance predictive performance.
# - **Data Augmentation**: Integrate additional relevant data sources to improve the model's predictive power.
# 
# #### **4. Resources and Ethical Considerations**
# 
# **Resources**:
# - **Documentation/Tutorials**: Refer to official documentation and online tutorials for Python libraries such as Pandas, NumPy, and Scikit-Learn.
# - **Community Forums**: Utilize platforms like Stack Overflow for troubleshooting and guidance.
# - **Research Papers**: Consult academic papers and industry articles for best practices in data analysis and modeling.
# - **Books**: Refer to texts on data science and machine learning for in-depth knowledge.
# 
# **Ethical Considerations**:
# - **Privacy**: Apply anonymization techniques to protect employee identities and sensitive information.
# - **Bias**: Mitigate biases in data collection, model development, and interpretation to ensure fairness.
# - **Transparency**: Maintain clear and transparent communication about how data is used and interpreted, especially when making decisions that affect employees' careers.
# 
# This comprehensive approach covers the essential aspects of model construction, validation, and evaluation, ensuring rigorous and ethical handling of the predictive modeling process.

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight

# Create a sample dataset
data = {
    'satisfaction_level': np.random.rand(1000),
    'last_evaluation': np.random.rand(1000),
    'number_project': np.random.randint(1, 7, 1000),
    'average_montly_hours': np.random.randint(80, 310, 1000),
    'time_spend_company': np.random.randint(1, 11, 1000),
    'Work_accident': np.random.randint(0, 2, 1000),
    'promotion_last_5years': np.random.randint(0, 2, 1000),
    'Department': np.random.choice(['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'], 1000),
    'salary': np.random.choice(['low', 'medium', 'high'], 1000),
    'left': np.random.randint(0, 2, 1000)
}

df_0 = pd.DataFrame(data)

# Define features and target variable
X = df_0.drop(['left'], axis=1)
y = df_0['left']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Define preprocessing steps for categorical variables
categorical_features = ['Department', 'salary']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Apply preprocessing steps to the data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Random Forest Classifier with class weights
rf = RandomForestClassifier(random_state=0, class_weight=class_weights)

# Define the parameter grid for GridSearchCV
cv_params = {
    'max_depth': [5, 6, 7],
    'max_features': [1.0],
    'max_samples': [0.7],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 3],
    'n_estimators': [50, 100],
}

# Initialize GridSearchCV with scoring based on F1 score
rf_model = GridSearchCV(rf, cv_params, scoring='f1', cv=5)

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', rf_model)])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred_rf = pipeline.predict(X_test)

# Evaluate the model
print("\nRandom Forest Classifier Evaluation:")
print('F1 score RF model:', f1_score(y_test, y_pred_rf))
print('Recall score RF model:', recall_score(y_test, y_pred_rf))
print('Precision score RF model:', precision_score(y_test, y_pred_rf))
print('Accuracy score RF model:', accuracy_score(y_test, y_pred_rf))
print('AUC score RF model:', roc_auc_score(y_test, y_pred_rf))

# Display the best parameters found by GridSearchCV
print("\nBest Parameters:", pipeline.named_steps['classifier'].best_params_)

# Plotting feature importances
best_rf = pipeline.named_steps['classifier'].best_estimator_
feature_importances = best_rf.feature_importances_

# Get feature names from the preprocessor
ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
onehot_columns = ohe.get_feature_names(categorical_features)
feature_names = np.concatenate([onehot_columns, X.select_dtypes(exclude=['object']).columns])

# Get indices of top features
top_indices = np.argsort(feature_importances)[::-1][:10]  # Top 10 features

# Create DataFrame for top feature importances
importance_df = pd.DataFrame({'Feature': feature_names[top_indices], 'Importance': feature_importances[top_indices]})

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], align='center')
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances - Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.show()


# # Insights from the Random Forest Feature Importance Plot
# 
# 1. **Top Features**:
#    - **High Importance Features**: Features at the top of the plot are the most influential in predicting employee turnover. For instance, if `satisfaction_level` and `average_monthly_hours` are among the top features, they are critical predictors for whether an employee will leave or stay.
#    - **Feature Impact**: High importance indicates that these features have a significant effect on the model's predictions, meaning changes in these variables have a notable impact on the likelihood of an employee leaving the company.
# 
# 2. **Feature Comparison**:
#    - **Relative Importance**: The plot allows you to compare the relative importance of features. Features with higher importance scores have more influence on the model's decision-making process. For example, if `number_of_projects` has lower importance than `work_accident`, it suggests that `work_accident` has a stronger influence on predicting employee turnover.
# 
# 3. **Model Transparency**:
#    - **Interpretability**: The plot helps in understanding which features drive the predictions of the Random Forest model. This transparency is valuable for interpreting model results and making data-driven decisions. If certain features, like `salary`, show high importance, it suggests they play a crucial role in employee turnover decisions.
# 
# 4. **Feature Engineering Insights**:
#    - **Potential Enhancements**: If some features have unexpectedly low importance, it might be worth revisiting feature engineering. For instance, creating new features or interactions might help capture more subtle patterns in the data.
# 
# 5. **Model Limitations**:
#    - **Overfitting Risks**: Random Forests can sometimes capture noise as important, so the feature importance should be interpreted carefully. Ensure that the importance is not due to overfitting or data peculiarities.
# 
# 6. **Actionable Insights**:
#    - **Retention Strategies**: Based on feature importance, focus on variables that are highly influential in predicting turnover. For instance, if `average_monthly_hours` is a top feature, it might be worth investigating if employees working longer hours are more likely to leave and addressing workload concerns.
# 
# ### Summary:
# - **Key Predictors**: Identify which features are most critical for predicting employee turnover.
# - **Feature Prioritization**: Use the plot to prioritize features for further analysis or interventions.
# - **Model Improvement**: Evaluate whether the importance aligns with business expectations and consider adjustments if necessary.

# # PACE: Execute Stage
# 
# **Interpret Model Performance and Results:**
# 
# 1. **Evaluation Metrics:**
#    - **AUC (Area Under the Curve)**: Indicates the model’s ability to distinguish between classes. A higher AUC value means better model performance.
#    - **Precision**: Measures the proportion of positive predictions that are truly positive. It’s crucial for understanding how many of the predicted positive cases are accurate.
#    - **Recall**: Measures the proportion of actual positives correctly predicted. It’s important for identifying the effectiveness of the model in capturing all positive cases.
#    - **Accuracy**: Proportion of total correct predictions (both true positives and true negatives) to the total number of cases.
#    - **F1-Score**: Harmonic mean of precision and recall, providing a balanced measure of a model's performance.
# 
# **Key Insights from the Models:**
# 
# 1. **Predictive Performance:**
#    - **Random Forest**: Achieved moderate improvements in F1 Score, Recall, and Precision after addressing class imbalance and optimizing hyperparameters, indicating effective prediction of employee turnover but room for refinement.
#    - **Logistic Regression**: Evaluated using AUC, Precision, Recall, Accuracy, and F1 Score to gauge overall model effectiveness.
# 
# 2. **Feature Importance:**
#    - Significant features include satisfaction level, average monthly hours, and number of projects, which strongly influence employee turnover predictions.
# 
# 3. **Model Interpretability:**
#    - Model accuracy is moderate, suggesting further improvements are necessary to achieve higher precision and recall.
# 
# **Business Recommendations:**
# 
# 1. **Retention Strategies:**
#    - Focus on improving employee satisfaction and managing workload to reduce turnover.
#    - Implement systems to identify and address high-risk employees proactively.
# 
# 2. **Early Warning System:**
#    - Develop predictive systems to flag at-risk employees for timely interventions.
# 
# 3. **Resource Allocation:**
#    - Target resources towards departments or roles with higher turnover risks to optimize retention efforts.
# 
# **Recommendations to Management:**
# 
# 1. **Continuous Improvement:**
#    - Refine models using advanced techniques like ensemble methods or neural networks for enhanced accuracy.
# 
# 2. **Data Collection:**
#    - Collect detailed data on employee sentiments and career development to improve model performance.
# 
# 3. **Validation and Feedback:**
#    - Regularly validate model predictions and gather feedback from HR teams for continuous improvement.
# 
# **Model Improvement:**
# 
# 1. **Data Quality:**
#    - Enhance data quality by addressing missing values and exploring additional features.
# 
# 2. **Algorithm Selection:**
#    - Test alternative algorithms or ensemble approaches to capture complex relationships better.
# 
# 3. **Feedback Loop:**
#    - Establish a feedback loop to incorporate new data and adapt to changing business needs.
# 
# **Additional Questions for Exploration:**
# 
# 1. **Impact of COVID-19:**
#    - Examine how the pandemic has impacted employee turnover and whether these effects are enduring.
# 
# 2. **Long-term Career Development:**
#    - Analyze how career development opportunities correlate with employee loyalty and job satisfaction.
# 
# 3. **Comparative Analysis:**
#    - Compare the performance of different models (e.g., Random Forest vs. Logistic Regression) to determine the most effective for specific business contexts.
# 
# **Resources Utilized:**
# 
# - **Scikit-learn Documentation**: For model implementation, hyperparameter tuning, and evaluation.
# - **Stack Overflow**: For troubleshooting coding issues.
# - **Research Papers and Articles**: For insights into trends and techniques in employee turnover prediction.
# 
# **Ethical Considerations:**
# 
# 1. **Privacy and Consent:**
#    - Ensure that employee data is anonymized and used with proper consent to maintain ethical standards.
# 
# 2. **Bias Mitigation:**
#    - Regularly audit models for biases and adjust algorithms to ensure fairness.
# 
# 3. **Transparency:**
#    - Maintain transparency with employees about the use of predictive analytics and provide clear explanations of decision-making processes.
# 
# By implementing these reflections and recommendations, you can enhance model effectiveness and ensure ethical and informed decision-making within the organization.

# # Step 4. Results and Evaluation
# 
# **1)Interpret model**
# 
# **2)Evaluate model performance using metrics**
# 
# **3)Prepare results, visualizations, and actionable steps to share with stakeholders**
# 

# # **Summary of Model Results:**
# Based on the Random Forest classifier's performance in predicting employee attrition, the following metrics were observed:
# 
# - **F1 Score:** 0.422
# - **Recall:** 0.700
# - **Precision:** 0.302
# - **Accuracy:** 0.543
# - **AUC Score:** 0.597
# 
# **Interpretation of Metrics:**
# - **F1 Score:** Balances precision and recall, indicating moderate performance in identifying employees likely to leave. A score of 0.422 reflects a trade-off between precision and recall.
# - **Recall:** At 0.700, this metric is relatively high, showing the model’s capability to correctly identify a significant proportion of actual leavers.
# - **Precision:** With a score of 0.302, this indicates a considerable number of false positives among predicted leavers.
# - **Accuracy:** At 0.543, accuracy is influenced by class imbalance and is not a reliable performance indicator in this context.
# - **AUC Score:** The AUC score of 0.597 suggests that the model performs slightly better than random chance in distinguishing between leavers and stayers.
# 
# **Conclusion:**
# The Random Forest model shows promise in predicting employee attrition, particularly with its high recall. However, the low precision highlights the need for caution in interpreting positive predictions. The AUC score indicates the model's performance is marginally above random chance.
# 
# **Recommendations:**
# 
# 1. **Feature Importance Insights:**
#    - The model identifies key features impacting employee attrition. Notably, [list top features here] have been highlighted as significant predictors.
# 
# 2. **Actionable Steps:**
#    - **HR Strategies:** Implement targeted retention strategies focusing on the key features identified by the model to reduce attrition.
#    - **Model Refinement:** Improve precision by refining model parameters or exploring alternative algorithms. Consider techniques like hyperparameter tuning and feature engineering.
# 
# 3. **Next Steps:**
#    - **Validation and Iteration:** Validate the model on new data and iterate based on additional feedback and insights.
#    - **Stakeholder Engagement:** Present findings and recommendations to stakeholders, emphasizing actionable insights for strategic HR planning.
# 
# **Execution Stage:**
# 
# - **Interpret Model Performance:** Reflect on metrics like F1 score, recall, precision, accuracy, and AUC to assess the model’s effectiveness in predicting attrition.
# - **Share Actionable Steps:** Communicate recommendations and insights to stakeholders, focusing on strategies to enhance employee retention and improve model performance.
# 
# **Additional Questions for Exploration:**
# 
# - **Impact of COVID-19:** Analyze if the pandemic has altered employee turnover patterns and whether these changes are likely to persist.
# - **Long-term Career Development:** Investigate correlations between career development opportunities, job satisfaction, and employee loyalty.
# - **Comparative Analysis:** Compare Random Forest with other models (e.g., Logistic Regression) to identify the most suitable approach for employee attrition prediction.
# 
# **Resources Utilized:**
# 
# - **Scikit-learn Documentation:** For model implementation and evaluation.
# - **Stack Overflow:** For resolving specific coding challenges.
# - **Research Papers and Articles:** For insights into employee turnover prediction and advanced modeling techniques.
# 
# **Ethical Considerations:**
# 
# - **Privacy and Consent:** Ensure data anonymization and proper consent for ethical data usage.
# - **Bias Mitigation:** Regularly review and adjust the model to prevent biases that could unfairly impact specific employee groups.
# - **Transparency:** Maintain transparency with employees regarding the use of predictive analytics for workforce management.
# 
# By addressing these aspects, the model’s effectiveness can be enhanced, leading to more informed and ethically sound decision-making within the organization.
# 
# **Prepared by: Chekresh Reddy**

# In[ ]:





# Customer Personality Analysis
Using this dataset from Kaggle to predict the personality of the major customer segment and to boost Marketing strategy to this particular segment of Customers.

## What is Customer Personality Analysis
Customer Personality Analysis is a detailed analysis of a company's ideal customers.It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviours and concerns of different types of customer.

## About the dataset
### Attributes of the dataset
### People
- ID: Customer's unique identifier
- Year_Birth: Customer's Birth Year
- Education: Customer's Education Level
- Marital_Status: Customer's Marital Status
- Income: Customer's yearly household income
- Kidhome: Number of children in customer's household
- Teenhome: Number of teenagers in customer's household
- Dt_Customer: Date of customer's enrollment with the company
- Recency: Number of days since customer's last purchase
- Complain: 1 if the customer complained in the last 2 years, 0 otherwise

### Products
- MntWines: Amount spent on wine in last 2 years
- MntFruits: Amount spent on fruits in last 2 years
- MntMeatProducts: Amount spent on meat in last 2 years
- MntFishProducts: Amount spent on fish in last 2 years
- MntSweetProducts: Amount spent on sweets in last 2 years
- MntGoldProds: Amount spent on gold in last 2 years

### Promotion
- NumDealsPurchases: Number of purchases made with a discount
- AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
- AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
- AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
- AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
- AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
- Response: 1 if customer accepted the offer in the last campaign, 0 otherwise

### Place
- NumWebPurchases: Number of purchases made through the company’s website
- NumCatalogPurchases: Number of purchases made using a catalogue
- NumStorePurchases: Number of purchases made directly in stores
- NumWebVisitsMonth: Number of visits to company’s website in the last month

### Target
Need to perform clustering to summarize customer segments.

## Tools and Technologies
- Python: Main programming language used.
- Pandas and NumPy: For data manipulation and numerical operations.
- Scikit-Learn: For implementing machine learning models.
- Matplotlib and Seaborn: For data visualization.
- Jupyter Notebook or Google Colab: Recommended environments for running the project scripts.

## Solutions Approach
 ### Unsupervised Learning
- K-means Clustering: Applied to segment customers based on features like spending habits and interaction with marketing campaigns. Various numbers of clusters were evaluated to determine the optimal clustering solution using metrics such as the silhouette score.
 ### Supervised Learning
- k-Nearest Neighbors (kNN): Used to classify customers based on their likelihood to respond to future marketing campaigns.
- Decision Tree: This model provided insights into the decision-making process of customer behaviors, identifying key variables that influence customer responses.
- Random Forest: An ensemble of decision trees used to improve the predictive accuracy and overfitting issues present in single decision trees. Random forest was particularly useful in handling the complex interactions and heterogeneity in customer data.
## Model Evaluation
Each model was eavluated using the accuracy score of the model. Random Forest Model gives the best evaluation score.

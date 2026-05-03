
### Title: Predicting Future Amazon Purchases Using Demographics and Purchase Histories

### Team
1.	Name: Alexa Lotano - POC <br>
    GitHub ID: alexa-lotano
2.	Name: Shahaan Khan - POC <br>
    GitHub ID: ShahaanK
3.	Name: Ben Euto - POC <br>
    GitHub ID: bene01-git

### Introduction (Not Done)

### Literature Review (not done – need to add stakeholder needs and why we chose our methods based on prior work and nature of the problem)
Previous approaches in analyzing consumer behavior in the e-commerce industry are rooted in machine learning techniques used to understand and create personalized recommendations based on user behavior, preferences and historical interactions[^1]. These algorithms use data from entire customer-bases to determine and recommend items frequently bought by users with similar buying patterns. Such algorithms utilize machine learning techniques including neural networks, classification, Naïve Bayes, decision trees, logistic regression, and clustering to analyze buying patterns and create accurate recommendations[^1] [^2]. Beyond analyzing buying patterns, previous research also includes sentiment analysis based on customer reviews of products, uncovering opinions of consumers and providing insight into future purchase or repurchase patterns[^2].

Limited research has been performed to understand the impact of demographics on e-commerce buying patterns. Al-Otaibi (2024) employed deep learning models to predict whether a customer is likely or not likely to buy an item based on age and income, in addition to buying patterns[^3]. However, Al-Otaibi (2024) suggested that the use of additional demographic information could provide deeper insights into what drives consumer purchase patterns, allowing for more accurate personalized recommendations[^3]

Limitations of current research and recommendations exist, particularly regarding the lack of demographic analysis in current systems. Demographics play a large role in user behavior, making it difficult for systems that lack such data to target niche groups. The result is a system that is too general to accurately predict user needs and distracts users with suggestions that do not fit their needs and desires. Incorporating demographic analysis into recommendations is vital to improving consumer engagement with e-commerce platforms and recommendations.

### Data and Methods

### Data (Not done – expand on where the data came from, how we know it is good data, and add visualizations)
The dataset used contains 5 years’ worth of crowdsourced Amazon purchase histories and their user demographics, spanning from 2018 to 2022. 

DOI: 10.7910/DVN/YGLYDY

The dataset contains information regarding participants' past Amazon purchases, as well as demographic and personal information. Data was collected from 5,027 unique survey participants and contains approximately 1,050,000 Amazon purchases from 2018–2022.

All data was collected from consenting Amazon users and does not contain any uniquely identifiable information about participants. 

Multiple CSV files were included:

amazon-purchases.csv
- Details about the Amazon orders themselves, including the order date, state the shipping address is located in, purchase price, quantity of product purchased, product name, and category product can be found in among other values
- Includes a column named SurveyResponsesID (randomly generated at the time of collection) which links the user's survey response to their Amazon purchase

survey.csv
- Survey responses including only responses from users who willingly chose to participate and share their data
- Also includes the SurveyResponseID column as a link with amazon-purchases.csv

fields.csv
- Names and descriptions of columns in survey.csv
- Fields/survey columns correspond to survey questions

Limitations of crowdsourced data: Because participants self-selected into the survey, the dataset may not be perfectly representative of all Amazon users. We note this as a limitation but believe the dataset is sufficiently large and diverse for our modeling purposes.

Due to the extremely large file sizes, we sampled 800 survey responses and will be analyzing only the Amazon purchases linked to these survey respondents. After cleaning and transforming the dataset, our data represents 157,026 Amazon purchases.

### Methods (Not Done)

**Preprocessing**
We began by cleaning and transforming data to prepare it for modeling. This process included:
1.	Data Integration and Subsetting: We joined amazon-purchases.csv with survey.csv using the SurveyResponseID field to create a singular dataset combining purchase behavior with demographic attributes. We then subsetted the data to only include purchases for a random selection of 800 survey participants to mitigate future memory and storage constraints.
2.	Temporal Feature Engineering: Order dates were parsed into day, month, and year variables.
3.	Categorical Encoding: Potential target variables, including product categories, titles, and ASIN/ISBN codes, were encoded using a label encoder. Remaining categorical variables were encoded using one-hot encoding, with one dummy variable from each variable removed to avoid redundancy.
4.	Handling Missing Data: Missing data was imputed when possible. This included mapping titles to their ASIN/ISBN code and using these mappings to fill in missing titles, as well as imputing “None” for missing survey data (i.e., when data was missing for respondents’ life changes, we assumed this meant there were no life changes to note). Remaining rows with missing data that could not be reliably imputed were subsequently removed from the dataset.
5.	Train/Test Split with Temporal Ordering: Since we are using temporal data, we used all data from before 2021 as our training data, and all data from 2021 onwards as our test data, as opposed to a traditional 80/20 split. This was done to preserve the temporal nature of our data and prevent data leakage.

While modeling, we performed additional data cleaning and transformation steps as needed for specific modeling attempts. These steps included additional temporal feature engineering and sequence construction, as well as grouping product categories. These steps will be described in relation to the modeling techniques they were implemented for in the following section.

**Modeling Techniques**
After exploring, cleaning, and preprocessing our data, we used numerous modeling techniques to understand patterns in the data and build a prediction model. Of our three possible target variables, we decided to attempt to predict only product categories. This is because ASIN/ISBN codes and product titles simply had too many unique values. We found that categories grouped products effectively enough that predicting categories would provide meaningful insight into possible recommendations for users. The goal of our recommendation model is to provide users with a few products that they may want to purchase based on their purchase history and demographics, not necessarily to predict their exact next purchase.

Our initial modeling techniques included:
1.	Dimensionality Reduction and K-Means Clustering: 
2.	Association Rule Mining:
3.	Bayesian Rule Mining:
4.	Random Forest:
5.	Neural Networks (including basic MLPs, Wide and Deep neural networks, and LSTMs):

While predicting what category of product a user might purchase next, we began to find that the number of unique categories in our data was still much too large, with over 1,600 unique categories. This not only limited our ability to accurately predict purchase categories, but also introduced memory issues and slowed down our models significantly. To mitigate this, we attempted to collapse our categories into a smaller number of more general categories. 

[Modeling attempts post category consolidation]

[demographic predictions]

**Evaluation Strategy**

### Supporting files (Not done – essentially an index)
### Results (Not Done)
### Discussion (Not Done)
### Limitations (Not Done)
### Future work (Not Done)

References
[^1]: Raji, M. et al., "E-commerce and consumer behavior: A review of AI-powered personalization and market trends." 2024.  
[^2]: Gupta, K. et al., "E-Commerce Customer Behavior Using Machine Learning." 2024.  
[^3]: Al-Otaibi, Y., "Enhancing e-Commerce Strategies: A Deep Learning Framework for Customer Behavior Prediction." 2024.
[^4] Aurélien Géron, "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, Third Edition", 2022.

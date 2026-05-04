
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

Expanding on previous research, we intend to test a variety of both previously explored methods, such as neural networks, as well as new techniques such as ensemble methods, including XGBoost and Random Forest models. We acknowledge the novelty of existing approaches, and hope that the additional demographic features in our data will allow us to improve on existing methods and models.

**Stakeholder Needs:**
Amazon Executives: Personalized recommendations can help Amazon increase sales, engagement, and conversion rates by simplifying the consumer experience. Additionally, platform personalization can contribute to favorable opinions about the company.

Consumers: The primary motivations for the use of e-commerce are convenience and wide product selections. However, the wide range of products available can make it difficult for users to quickly find the items they need. Creating the most accurate recommendations possible for individual users can alleviate decision fatigue and endless searching for the right products.

Sellers: Third party sellers will benefit from our system as their products will be promoted to the customers who are most likely to purchase their products.


### Data and Methods

### Data (Not done – Add visualizations)
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

This data came from the Harvard Dataverse, which is Harvard University's research data repository. Harvard University is renowned for their research output, and they are consistently ranked the #1 research university in world each year, including this past year[^5]. This data also only consists of those who consented to participate in the survey, ensuring ethical collection that does not invade anyone's private data. With all of this in mind,, we believe this is legitimate and credible data that will be of great use for our analysis.

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
1.	Dimensionality Reduction and KNN Classification: Our first attempt at modeling our data consisted of dimensionality reduction and subsequent K-nearest neighbor classification. We attempted a few different dimensionality reduction techniques, including Principal Component Analysis (PCA), Factor Analysis, and UMAP. Using PCA, we were able to capture 95 percent of variance in our data using 93 factors. Factor analysis was assessed using a KMO model, which indicated that it was unsuitable for our dataset. UMAP was attempted with 2 components. We then performed KNN classification on both the PCA and UMAP reduced datasets, as well as the original dataset. We found that the PCA reduced data was classified more accurately than the UMAP reduced data, but both reduction methods led to significantly worse classification results than the original data. Thus, we abandoned dimensionality reduction methods, and moved forward with our original dataset.
2.	Association Rule Mining: Due to poor classification results in our KNN classification model, we decided to try association rule mining to get a better understanding of the data and items that are frequently purchased by the same customer based on survey response ID. It is important to note that this method required us to use only survey response ID and the categories, so it does not give us a comprehensive understanding of the impact of demographics on purchase behavior, or other factors such as purchase price, quantity, order date, or sequential features such as time between orders. This model served merely to give us a better understanding of relationships between product categories, and which categories are often purchased by the same customers.
3.	Bayesian Rule Mining: Per our professor's suggestion, we performed Bayesian rule mining to evaluate and refine discovered rules about customer purchasing behavior across product categories. We read in an unencoded copy of the dataset to facilitate the process, computed Bayesian-adjusted confidence scores using Laplace smoothing, and calculated the Kemeny–Oppenheim confirmation measure to quantify the strength of evidence for each rule. We then  used pgmpy's Hill Climb search with BIC scoring to develop a Bayesian Network structure over frequent product categories, fit the network's parameters to display conditional probability tables (CPTs), visualized the directed dependency graph with NetworkX, surfaced the top rules filtered based on their Kemeny–Oppenheim measures, and added a category connectivity bar chart showing in-degree and out-degree per node. The resulting analysis and model identified category sets with high support and confidence grounded in both observed evidence and probabilistic priors, ultimately updating beliefs about which categories of items are commonly purchased together and verifying (or potentially disputing) earlier association rule mining findings to better inform cross-category customer recommendations.
4.	Random Forest:
5.	Neural Networks:
    Next, we attempted various neural networks to predict categories. We began with a simple multilayer perceptron (MLP) with 2 hidden layers. We compiled the model using a loss function of sparse categorical cross entropy and an SGD optimizer, and using accuracy as our metric. We then attempted a wide and deep neural network, subsetting features so that the wide and deep components were trained on different features. This meant features that were likely linearly related to categories were used to train the wide component, and remaining features where interactions were likely present were used to train the deep component. At this point, we also examined the correlations between variables and removed some variables that were extremely highly correlated to other variables. We predicted that doing so would help mitigate memory constraints and simplify our models without sacrificing accuracy. Our wide and deep neural network performed significantly worse than our original MLP, so this was abandoned. We tested the MLP again after removing the highly correlated values to confirm that accuracy was not greatly impacted. 

    Next, we attempted to build a Long-Short Term Memory (LSTM) network. At this point, we conducted sequential feature engineering to create new variables in our dataset, allowing the LSTM to identify sequential patterns. These features included days since last purchase, and the position in the customers order sequence. We also kept the survey response ID variable when training the LSTM to ensure that different customers were being treated as such. Although the LSTM did not improve on the accuracy of our previous model, we tested our original MLP on the data with the new features, which resulted in a slight improvement.

    We attempted numerous hyperparameter combinations with our MLP, including number of layers, number of nodes per layer, activation functions, optimizers, and callbacks. We found that our best model was the MLP after removing highly correlated features and adding sequential and time-based features, with the original hyperparameters.


While predicting what category of product a user might purchase next, we began to find that the number of unique categories in our data was still much too large, with over 1,600 unique categories. This not only limited our ability to accurately predict purchase categories, but also introduced memory issues and slowed down our models significantly. To mitigate this, we attempted to collapse our categories into a smaller number of more general categories. 

[Modeling attempts post category consolidation]

[demographic predictions]

[identify and state best model]

**Evaluation Strategy**

[identify top few models (possibly one neural network and our random forest and xgboost models) and evaluate beyond just accuracy]

### Supporting files (Not done – essentially an index - add brief descriptions of each)
Our supporting files located in the work folder are as follows:
1. 01-purchase-data-eda.ipynb
2. 02-survey-data-eda.ipynb
3. 03-data-merge-and-cleaning.ipynb
4. 04-data-transformation.ipynb
5. 05-dimensionality-reduction-and-visualization.ipynb
6. 06-modeling.ipynb
7. 07-association-rule-mining.ipynb
8. 08-bayesian-rule-mining.ipynb
9. 09-random-forest.ipynb
    - 09-random-forest.py
10. 10-neural-networks.ipynb
11. 11-xgboost.ipynb
    - 11-xgboost.py
12. 12-lstm.ipynb
13. 13-neural-network-tuning.ipynb
14. 14-demographic-prediction.ipynb
15. 15-category-consolidation.ipynb
16. 16-final-evaluation.ipynb

### Results (Not Done)
### Discussion (Not Done)
### Limitations (Not Done)
### Future Work (Not Done)

References
[^1]: Raji, M. et al., "E-commerce and consumer behavior: A review of AI-powered personalization and market trends." 2024.  
[^2]: Gupta, K. et al., "E-Commerce Customer Behavior Using Machine Learning." 2024.  
[^3]: Al-Otaibi, Y., "Enhancing e-Commerce Strategies: A Deep Learning Framework for Customer Behavior Prediction." 2024.
[^4] Aurélien Géron, "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, Third Edition", 2022.
[^5] Voronoi. (2025, July 17). Harvard University is the top research university of 2025. https://www.voronoiapp.com/education/Harvard-University-is-the-Top-Research-University-of-2025--5816

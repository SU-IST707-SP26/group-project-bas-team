
### Title: Predicting Future Amazon Purchases Using Demographics and Purchase Histories

### Team
1. Name: Alexa Lotano - POC <br>
    GitHub ID: alexa-lotano
2. Name: Shahaan Khan - POC <br>
    GitHub ID: ShahaanK
3. Name: Ben Euto - POC <br>
    GitHub ID: bene01-git

### Introduction
For this project, we wish to examine customer behavior when it comes to e-commerce websites, mainly Amazon. Amazon is the world's biggest online retailer, with millions of active users worldwide and billions of monthly visits. With this in mind, we feel that it is worth taking a closer look at what users shop for, what might influence what they shop for, and what information could be revealed about the different kinds of users that populate Amazon's services.

**What are we trying to do?**
We are trying to use current Amazon customer activity to predict future Amazon customer activity. For example, if a user buys a set of pans, a cutting board, and oven mitts, would we have reason to believe that they would buy another kitchen utensil or appliance (i.e. a cookie sheet or a casserole dish)? Would we have reason to believe they are trying to cook more (or learn how to)? We plan to do this by examining customers' demographics, how much of a certain item they may buy, where they are receiving the item (location can certainly be a factor; you could order a new jacket if you live in a cold state or you could order sunscreen if you live in a hot state), and the type of product (in other words, the category) they are looking for.

**What's new in our approach and why do we think it would be successful?**
We believe that the novel part of our approach is our focus on future customer behavior as opposed to current customer behavior. Instead of examining what types of products customers tend to gravitate towards, we are focusing on what types of products customers could gravitate towards next. Additionally, we will examine the role of demographics in customer buying patterns and use our findings to anticipate purchasing activity of more niche groups of Amazon users. Each individual person has their own story and the products they purchase are a strong indicator of said stories. But what could the next chapter of their stories be, and how would that be a reflection of what they currently purchase? We believe this approach will be successful because it could provide useful insights about how customer needs shift and what products could be further promoted to adjust to that shift. Keeping up with the times is important to any business, but it is especially important to a retailer as large as Amazon, even moreso as an online business.

**Who cares? If we are successful, what difference could we make?**
We believe the people who would care the most are Amazon's stakeholders. If we are successful, it could improve the business model of Amazon and strengthen their algorithms to the point where they do not just recommend products based on what one is currently buying (i.e. offering similar products), but take the bigger picture into account and recommend products based on what they might look for next (i.e. if someone buys a phone, maybe recommend them a phone case, or a charger, or maybe even earbuds). Incorporating demographic data into our system with further strengtehn Amazon's business model by enhancing recommendations and increasing engagement with recommendations. This adjustment to Amazon's business model can change the way online shopping is done and once again help Amazon pioneer the e-commerce market, resulting in a net gain for them and their stakeholders.

### Literature Review
Previous approaches in analyzing consumer behavior in the e-commerce industry are rooted in machine learning techniques used to understand and create personalized recommendations based on user behavior, preferences and historical interactions[^1]. These algorithms use data from entire customer-bases to determine and recommend items frequently bought by users with similar buying patterns. Such algorithms utilize machine learning techniques including neural networks, classification, Naïve Bayes, decision trees, logistic regression, and clustering to analyze buying patterns and create accurate recommendations[^1] [^2]. Beyond analyzing buying patterns, previous research also includes sentiment analysis based on customer reviews of products, uncovering opinions of consumers and providing insight into future purchase or repurchase patterns[^2].

Limited research has been performed to understand the impact of demographics on e-commerce buying patterns. Al-Otaibi (2024) employed deep learning models to predict whether a customer is likely or not likely to buy an item based on age and income, in addition to buying patterns[^3]. However, Al-Otaibi (2024) suggested that the use of additional demographic information could provide deeper insights into what drives consumer purchase patterns, allowing for more accurate personalized recommendations[^3]

Limitations of current research and recommendations exist, particularly regarding the lack of demographic analysis in current systems. Demographics play a large role in user behavior, making it difficult for systems that lack such data to target niche groups. The result is a system that is too general to accurately predict user needs and distracts users with suggestions that do not fit their needs and desires. Incorporating demographic analysis into recommendations is vital to improving consumer engagement with e-commerce platforms and recommendations.

#### Stakeholder Needs:

Consumers: The primary motivations for the use of e-commerce are convenience and wide product selections. However, the wide range of products available can make it difficult for users to quickly find the items they need. Creating the most accurate recommendations possible for individual users can alleviate decision fatigue and endless searching for the right products.

Amazon Executives: Personalized recommendations can help Amazon increase sales, engagement, and conversion rates by simplifying the consumer experience. Additionally, platform personalization can contribute to favorable opinions about the company.

Sellers: Third party sellers will benefit from our system as their products will be promoted to the customers who are most likely to purchase their products.


### Data and Methods

#### Data
The dataset we will be using contains 5 years worth of crowdsource Amazon purchase histories and their user demographics, spanning from 2018 to 2022. The variables relating to user demographics were collected through an online survey. The dataset also includes details about the questions used in the survey and the format they were presented in. All data collected was exported by Amazon users from the Amazon website and were shared by consenting users. Multiple CSV files are included:

**amazon-purchases.csv**
- Details about the Amazon orders themselves, including the order date, state the shipping address is located in, purchase price, quantity of product purchased, product name, and category product can be found in among other values
- Also includes a column named SurveyResponsesID (randomly generated at the time of collection) which links the user's survey response to their Amazon purchase

**survey.csv**
- Survey responses including only responses from users who willingly chose to participate and share their data
- Also includes the SurveyResponseID column as a link with amazon-purchases.csv

**fields.csv**
- Names and descriptions of columns in survey.csv
- Fields/survey columns correspond to survey questions


#### Methods

As we continue to work through this course, we may find these approaches to not be optimal and as such are subject to change when better solutions or methodology is found.

**Preprocessing and Transformations**

Our preprocessing will involve several cleaning transformations to prepare the data for modeling:

1. **Data Integration**: We will join `amazon-purchases.csv` with `survey.csv` using the `SurveyResponseID` field to create a singular dataset combining purchase behavior with demographic attributes.

2. **Temporal Feature Engineering**: Order dates will be parsed and used to construct sequential purchase histories for each user. We will create time-based features including purchase frequency, time between purchases, and seasonal purchasing patterns.

3. **Categorical Encoding**: Product categories will be encoded using label encoding or one-hot encoding depending on the model requirements. Demographic variables (age groups, income brackets, location) will similarly be transformed into numerical representations. We will check for which will yield the best results.

4. **Sequence Construction**: For each user, we will construct ordered sequences of product categories purchased over time. These sequences will serve as input for our temporal models, with the target being the next category in the sequence.

5. **Handling Missing Data**: We will assess the extent of missing values in both purchase and survey data and apply imputation strategies or remove the data based on its nature.

6. **Train/Test Split with Temporal Ordering**: To simulate real-world conditions, we will use earlier purchases for training and later purchases for testing, preserving the temporal structure of the data, instead of the traditional 80/20 split.


**Modeling Techniques**

We are hoping to use a multi-pronged modeling approach to capture different aspects of purchasing behavior. The best modeling technique will be used at the end. As the course progresses, we may learn more optimal modeling techniques that will be added or exchanged with the ones listed below:

1. **Association Rule Mining (Apriori/FP-Growth)**: To identify frequently co-purchased product categories and establish baseline association patterns that indicate what products tend to follow others.

2. **Sequential Pattern Mining**: To discover common purchase sequences across users and identify temporal patterns in buying behavior.

3. **Classification Models**: We will train Random Forest and Gradient Boosting (XGBoost) classifiers to predict the next product category a user is likely to purchase, using demographic features and recent purchase history as inputs.

4. **LSTM Networks**: Long Short-Term Memory networks will be explored to model the sequential nature of purchase histories and capture long-term dependencies in buying patterns.

5. **Wide & Deep Neural Networks**: We will implement a Wide & Deep neural network architecture, which was originally developed for recommendation systems, using the "wide" and "deep" paths to memorize direct associations between demographic features and purchase patterns while the "deep" path learns abstract, non-obvious relationships through multiple hidden layers. (Chapter 10 introduces this approach)[^4]

6. **Clustering for Customer Segmentation**: K-Means or hierarchical clustering will be applied to demographic and behavioral features to identify distinct customer segments, allowing for segment-specific prediction models.

**Evaluation Strategy**

With stakeholders in mind we plan to conduct the following types of evalutation to ensure our results are of the highest accuracy and recommendations are truly useful:

- **Accuracy Metrics**: We will use precision, recall, and F1-score to evaluate next-category prediction performance, with focus on precision as this our key finding.

- **Top-K Accuracy**: We will measure whether the true next purchase appears in the top 3 or top 5 predicted categories.

- **Segment-Level Analysis**: We will evaluate model performance across different demographic segments to ensure our system serves diverse user groups equitably and effectively.

- **Cross-Validation**: K-fold cross-validation will be employed to ensure model robustness and prevent overfitting.

### Project Plan
| Period | Activity | Milestone |
|--------|----------|-----------|
| 1/26 - 2/8 | Stakeholder analysis and requirements gathering. Initial data exploration and quality assessment. Begin preprocessing. | Completed stakeholder analysis. Data quality report generated. Additional datasets identified as necessary. |
| 2/9 - 2/22 | Data preprocessing. Feature engineering for temporal and demographic variables. Sequence construction for users. | Unified dataset created. Feature set finalized. Purchase sequences constructed for all users. |
| 2/23 - 3/8 | Association rule mining implementation. Initial classification model development (Random Forest baseline). Begin customer segmentation clustering. | Baseline association rules identified. Initial classifier trained with preliminary accuracy metrics. Customer segments defined. |
| 3/9 - 3/22 | LSTM model development and training. Hyperparameter tuning for classification models. Comparative analysis across modeling approaches. Midterm report developed | LSTM model operational. Candidate modeling approaches finalized based on performance comparison. Midterm report submitted |
| 3/23 - 4/5 | Model refinement and optimization. Segment-specific model evaluation. Begin integration of demographic features into best-performing models. | Optimized models with demographic integration. Performance metrics documented across customer segments. |
| 4/6 - 4/19 | Final model evaluation and validation. Results interpretation and visualization. Documentation and report writing. | Final model selected. Comprehensive evaluation complete. Draft report and visualizations ready. |
| 4/20 - 5/3 | Final presentation preparation. Code cleanup and repository organization. Final report completion and submission. | Project deliverables complete. Final presentation delivered. All documentation and code submitted. |


### Risks
Our system faces risks associated with the collection and use of demographic data, particularly the ethical collection and use of such data. Unethical data collection and use can contribute to prejudiced or biased recommendation algorithms, as well as a lack of trust amongst consumers[^2]. Mitigating bias and ensuring ethical collection of demographic data is crucial to create an accurate and trusted model to enhance personalized recommendations on e-commerce platforms.

We also face a risk of overfitting our machine learning models to our data by targeting extremely niche groups of consumers. To mitigate this, we will utilize validation methods such as training and test sets and cross validation to evaluate our model.


### References
[^1]: Raji, M. et al., "E-commerce and consumer behavior: A review of AI-powered personalization and market trends." 2024.  
[^2]: Gupta, K. et al., "E-Commerce Customer Behavior Using Machine Learning." 2024.  
[^3]: Al-Otaibi, Y., "Enhancing e-Commerce Strategies: A Deep Learning Framework for Customer Behavior Prediction." 2024.
[^4] Aurélien Géron, "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, Third Edition", 2022.
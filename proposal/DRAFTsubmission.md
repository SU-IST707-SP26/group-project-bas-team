
### Title


### Team



### Introduction
For this project, we wish to examine customer behavior when it comes to e-commerce websites, mainly Amazon. Amazon, the world's biggest online retailer, has millions of active users worldwide and billions of monthly visits. With this in mind, we feel that it is worth taking a closer look at what users shop for, what might factor into them shopping for what they shop for, and what information could be revealed about the different kinds of users that populate Amazon's services.

**What are we trying to do?**
We are trying to use current Amazon customer activity to predict future Amazon customer activity. For example, if a user buys a set of pans, a cutting board, and oven mitts, would we have reason to believe that they would buy another kitchen utensil or appliance (i.e. a cookie sheet or a casserole dish)? Would we have reason to believe they are trying to cook more (or learn how to)? We plan to do this by examining customers' demographics, how much of a certain item they may buy, where they are receiving the item (location can certainly be a factor; you could order a new jacket if you live in a cold state or you could order sunscreen if you live in a hot state), and the type of product (in other words, the category) they are looking for.

**What's new in our approach and why do we think it would be successful?**
We believe that the novel part of our approach is our focus on future customer behavior as opposed to current customer behavior. Instead of examining what types of products customers tend to gravitate towards, we are focusing on what types of products customers could gravitate towards next. Each individual person has their own story and the products they purchase are a strong indicator of said stories. But what could the next chapter of their stories be, and how would that be a reflection of what they currently purchase? We believe this approach will be successful because it could provide useful insights about how customer needs shift and what products to further promote to adjust to that shift. Keeping up with the times is important to any business, but it is especially important to a retailer as large as Amazon, even moreso as an online business.

**Who cares? If we are successful, what difference could we make?**
We believe the people who would care the most are Amazon's stakeholders. If we are successful, it could improve the business model of Amazon and strengthen their algorithms to the point where they do not just recommend products based on what one is currently buying (i.e. offering similar products), but take the bigger picture into account and recommend products based on what they might look for next (i.e. if someone buys a phone, maybe recommend them a phone case, or a charger, or maybe even earbuds). This adjustment to Amazon's business model can change the way online shopping is done and once again help Amazon pioneer the e-commerce market, resulting in a net gain for them and their stakeholders.

### Literature Review


### Data and Methods



#### Data
The dataset we will be using contains 5 years' worth of crowdsource Amazon purchase histories and their user demographics, spanning from 2018 to 2022. The variables relating to user demographics were collected through an online survey. The dataset also includes details about the questions used in the survey and the format they were presented in. All data collected was exported by Amazon users from the Amazon website and were shared by consenting users. Multiple CSV files are included:

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



### Project Plan



### Risks



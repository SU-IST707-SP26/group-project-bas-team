# WORKLOG.md

## 2026-03-22 - Began Bayesian rule mining (Ben, Shahaan)
**Context:** Attempting Bayesian rule mining based off association rule mining as per professor suggestion

**Work Completed:**
- (Ben) - Read in copy of the data before encoding steps to facilitate Bayesian rule mining
- (Ben) - Began Bayesian rule mining attempts to identtify item categories that are commonly purchased by the same customer based on prior knowledge or probabilistic beliefs, essentially updating beliefs based on evidence+
- (Shahaan) Fixed data path from Codespaces to relative path for local development in vscode and created requirements.txt for virtual environment setup
- (Shahaan) Added Bayesian Network graph visualization using networkx
- (Shahaan) Added display of top Bayesian rules filtered by Kemeny-Oppenheim measure
- (Shahaan) Fit Bayesian Network parameters and displayed conditional probability tables (CPTs)
- (Shahaan) Added category connectivity bar chart showing in/out degree per category

**Files Created**
- work/08-bayesian-rule-mining.ipynb

**Impact:** Identified category sets with high support and/or high confidence based on existing evidence found through association rule mining and probabilistic beliefs, ultimately updating beliefs on what categories of items are often purchased by the same customer and verifying or potentially disputing prior evidence regarding what to recommend customers based on what they purchase.

**Next Steps:** Potentially continuing Bayesian rule mining efforts and examining additional measures as well as discussing possibility of obtaining an Azure instance during next team meeting.

## 2026-03-22 - Began association rule mining (Alexa)
**Context:** Attempting association rule mining as per professor suggestion

**Work Completed:**
- (Alexa) Saved a copy of the data before encoding steps to facilitate association rule mining 
- (Alexa) Began association rule mining attempts to identify item categories that are commonly purchased by the same customer

**Files Created:**
- work/07-association-rule-mining.ipynb
- data/data_unencoded

**Impact:** Identified category sets with high support and/or high confidence, which will help us identify what categories of items are often purchased by the same customer. This will serve as a starting point for recommendations by helping us understand if a customer purchases products from one category, which category we might recommend to them next.

**Next Steps**: Continue association mining efforts and examine additional measures. Continue classification attempts. Discuss possibility of obtaining an Azure instance in order to mitigate memory issues.

## 2026-03-06 - Midterm Checkpoint Submission (Team)
**Context:** Finalizing and submitting the Checkpoint 2 midterm report

**Work Completed:**
- (Team) Completed midterm checkpoint submission notebook
- (Shahaan) Added EDA visualizations to submission: purchase price distribution, purchase volume by year, age × income heatmap, top 15 product categories
- (Shahaan) Fixed data leakage in StandardScaler and PCA. Now fit on training data only, transform applied to test
- (Alexa) Compared KNN accuracy across reduction methods (no reduction: 3.38%, PCA: 1.50%, UMAP: 0.96%)
- (Team) Attempted Decision Tree and Random Forest classifiers — encountered memory allocation errors (~3.3 GB) due to dataset size (157K rows, 93 components, 1,600+ categories)
- (Team) Updated Overview section to address instructor feedback on narrowing project scope
- (Team) Converted preprocessing code cells to markdown code fences (reference only, not executable)
- (Team) Added Problems & Challenges section documenting memory constraints, data leakage fix, and ethics considerations
- (Team) Added Next Steps with week-by-week timeline through final report (5/5)

**Impact:** Midterm checkpoint submitted. Baseline KNN model established. Key finding: demographics alone are insufficient to predict product categories, sequential purchase features needed. Memory constraints identified as primary blocker for tree-based models.

**Next Steps**: Define prediction window. Begin sequential feature engineering (purchase sequences per user, time between purchases, category frequency). Consolidate rare categories to reduce target cardinality and address memory issues.

## 2026-03-04 - Dimensionality Reduction, Modeling, and Midterm Checkpoint (Alexa)
**Context:** Completed dimensionality reduction efforts and began modeling the data

**Work Completed:**
- (Alexa) Assessed suitability of factor analysis and attempted UMAP to reduce dimensionality, and compared results to PCA
- (Alexa) Fit a simple K-nearest neighbors classifier to the reduced data and assessed performance
- (Alexa) Contributed to midterm checkpoint submission file

**Files Created:**
- work/06-modeling

**Impact:** M4.T1, M4.T2, and M5.T1 complete. Dimensionality reduction complete and modeling efforts begun. Preprocessing, Modeling, and Problems & Challenges sections complete in midterm checkpoint submission, with Data section nearly complete.

**Next Steps**: Continue modeling data and complete midterm checkpoint submission.

## 2026-03-02 - Completed Data Transformation and Began Dimensionality Reduction (Alexa)
**Context:** Finished encoding data, began dimensionality reduction and data visualization efforts

**Work Completed:**
- (Alexa) Data transformations completed with revisions to data encoding
- (Alexa) Began dimensionality reduction using PCA and visualization of reduced data
- (Alexa) Sampled original survey data to create smaller dataset to mitigate kernel crashes

**Files Created:**
- work/04-data-transformation
- renamed: work/05-dimensionality-reduction-and-visualization.ipynb

**Impact:** Limited data significantly. Data transformations complete and dimensionality reduction begun.

**Next Steps**: Complete dimensionality reduction efforts and begin modeling

## 2026-03-01 - Dimensionality Reduction and Data Visualization (Team)
**Context:** Updating admin files, cleaning and transforming data, and beginning dimensionality reduction and visualization work

**Work Completed:**
- (Alexa) Finished cleaning dataset, including imputing missing data, and renaming columns
- (Alexa) Began data transformation, including exploding multi-value columns, converting data types, and feature engineering
- (Ben) StandardScaler and PCA inititated for future use+
- (Shahaan) Dropped irrelevant features, added ordinal encoding to ordered categorical columns, applied binary encoding to Yes/No columns, applied label encoding to nominal categorical columns, and left comment on to save cleaned dataset to `data/cleaned_data.pkl` and `data/cleaned_data.csv`

**Files Created:**
- work/04-dimensionality-reduction-and-visualization.ipynb
- checkpoint/submission.ipynb

**Impact:** Dataset nearly fully cleaned. Began data transformation and feature engineering. Code for dimensionality reduction set up.

**Next Steps**: Perform dimensionality reduction and visualize dataset. Begin modeling data.

## 2026-02-25 - Fourth Team Meeting and Check-in (Team)
**Context:** Updating admin files, checking in with teammates, and settings goals for following week

**Work Completed:**
- (Team) Set clear goals for the next week, including beginning to work more in-depth with data

**Impact:** Survey data read in, EDA performed. Dimensionality reduction started.

**Next Steps**: Finish cleaning dataset. Begin dimensionality reduction and visualization work.

## 2026-02-22 – Updated EDA (Alexa and Ben)

**Context:** Merging the purchase and survey data and early data cleaning steps

**Work Completed:**
- (Alexa) Merge the purchase and survey datasets
- (Alexa) Remove rows with excessing missing data
- (Alexa) Begin imputing missing values
- (Ben) Switched from imputing to dropping NAs as advised by team. Performed visualizations

**Files Created:**
- work/03-data-merge-and-cleaning.ipynb

**Impact:** M3.T1 completed. Merged datasets into a single dataframe and began imputing categorical data.

**Next Steps**: Continue data cleaning and preparation (imputing nulls, scaling numeric data, etc.)

## 2026-02-20 – Third Team Meeting and Proposal Creation (Team)

**Context:** Updating admin files and checking in with teammates

**Work Completed:**
- (Team) Set clear goals for the next week, including tweaking EDA

**Impact:** Data read in.

**Next Steps**: Potentially exploring other datasets.

## 2026-02-15 – Dataset Upload and Exploratory Data Analysis (Alexa and Ben)

**Context:** Exploratory analysis and visualization of raw data before cleaning and merging

**Work Completed:**
- (Alexa) Upload and read amazon purchase data and survey data
- (Ben) Explore and visualize raw amazon purchase data
- (Alexa) Explore and visualize raw survey data. Identify nulls and generate ideas to prepare the data for analysis

**Files Created:**
- work/01-purchase-data-eda.ipynb
- work/02-survey-data-eda.ipynb

**Impact:** M2.T1 and M2.T2 completed. Explored raw datasets to understand the data and identified necessary cleaning and preparation steps.

**Next Steps**: Merge amazon purchase data and survey data. Begin cleaning the data and preparing it for analysis (imputing nulls, scaling numeric data, etc.)


## 2026-02-08 – VISION.md and WORKPLAN.md Updated (Alexa)

**Context:** Updating admin files to create clear project plan and goals

**Work Completed:**
- (Alexa) Built project vision, including identifying stakeholders, problem statement, and proposed solution.
- (Alexa) Began identifying milestones and tasks to build project plan

**Impact:** Initial vision created. Initial workplan created.

**Next Steps**: Awaiting approval of proposal. Read data into project repository and began EDA.

## 2026-02-01 – Second Team Meeting and Proposal Creation (Team)

**Context:** Identifying data, organizing repository, and submitting proposal

**Work Completed:**
- (Team) Identified a dataset that will be used to create project
- (Alexa) Created VISION.md, WORKLOG.md, and WORKPLAN.md files in project repository
- (Team) Created and submitted project proposal

**Files Created:**
- proposal/submission.md
- admin/VISION.md
- admin/WORKLOG.md
- admin/WORKPLAN.md

**Impact:** M1.T2 and M1.T3 complete. Dataset identified. Proposal created and submitted. Project infrastructure in place.

**Next Steps**: Update vision and work plan. Await proposal approval.

## 2026-01-27 - Project Kickoff (Team)

**Context:** First team meeting to identify a project idea.

**Work Completed:**
- (Team) Determined an idea for a machine learning project
- Team members to individually search for appropriate datasets and share with team

**Impact:** M1.T1 complete. Project idea decided.

**Next Steps**: Individual team members search for datasets. Work on project proposal.
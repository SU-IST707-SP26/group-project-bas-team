# WORKLOG.md

## 2026-05-03 - Continue Working on Final Report (Alexa)
**Context:** Continue working on final report, including Data and Methods, Literature Review, and Supporting Files sections.

**Work Completed:**
- (Alexa) Cleaned up work files to remove excessive output and include clear titles in all files.
- (Alexa) Continued writing final report, including Data and Methods, Literature Review, and Supporting Files sections.
- (Alexa) Attempted to consolidate categories variable, but kernel kept crashing.

**Files Created:**
- work/15-category-consolidation.ipynb
- work/16-final-evaluation.ipynb

**Impact:** Literature Review section in final report complete. Data and Methods and Supporting Files sections nearly complete. Unable to complete category consolidation. Other team members will attempt.

**Next Steps:** Consolidate categories and re-run best models to see if accuracy improves. Conduct final evaluation of models. Continue writing final report.

## 2026-05-02 - Built Model to Predict Demographics and Began Work on Final Report (Alexa)
**Context:** Predicting demographics from purchase behavior and working on final report submission.

**Work Completed:**
- (Alexa) Built a multi-output neural network to predict demographics from purchase behavior. Model predicted all demographic variables better or the same as the most common value of each variable.
- (Alexa) Built outline and began writing final report, including Literature Review and Data and Methods sections.
- (Alexa) Updated workplan to better align with final goals and tasks.

**Impact:** Demographic prediction model successfully built, allowing our purchase behavior predictions to be extrapolated beyond our dataset.

**Next Steps:** Continue improving on purchase behavior prediction model and continue working on final report.

## 2026-04-26 - Random Forest Azure Migration (Shahaan)
**Context:** Moving Random Forest training to Azure ML to overcome local memory limitations. Full-data training (113K rows, 1,625 classes) OOMs on local machine and on the original F4s_v2 instance (8GB RAM) due to class probability storage at each tree node (13KB per node × 100 trees).

**Work Completed:**
- (Shahaan) Converted notebook 09 to a standalone Python script (`work/09-random-forest.py`) with headless plotting, timing logs, and fixed tuned model evaluation (was silently OOMing in notebook cell 22)
- (Shahaan) Diagnosed F4s_v2 (8GB) as insufficient for full RF on this dataset; provisioned Standard_E4s_v3 (4 cores, 32GB RAM) compute instance `szkhan2` in Azure ML
- (Shahaan) Cloned repo and started full-data RF training run on `szkhan2` — in progress

**Files Created:**
- work/09-random-forest.py

**Impact:** M5.T4 — Full-data Random Forest training now running on Azure with 32GB RAM. Results pending.

**Next Steps:** Collect RF results (accuracy, weighted F1, CV scores) from Azure run. Apply same Azure approach to XGBoost (M5.T5).

## 2026-04-20 - Final Neural Network Hyperparameter Tuning (Alexa)
**Context:** Ran previously built code to tune neural network hyperparameters

**Work Completed:**
- (Alexa) Revised code in all neural network notebooks to remove categories that are not present in the training set from the test set prior to training and testing and reran code. Improved training accuracy of best model (training accuracy=11.5%, test accuracy=7.1%)
- (Alexa) Ran code to test different optimizers and regularization techniques. Hyperparameter tuning did not improve accuracy.

**Impact:** Best accuracy is currently 7.1%.

**Next Steps:** Begin work on final project presentation and report.

## 2026-04-19 - Final Neural Network Hyperparameter Tuning (Alexa)
**Context:** Tested various optimizers and regularization techniques to attempt to improve existing neural networks.

**Work Completed:**
- (Alexa) Recreated our best performing neural network and built code to test different optimizers, including RMSprop, Adagrad, Adamax, Nadam, and AdamW.
- (Alexa) Began implementing various regularization techniques, including L1, L2, Dropout, and Max-Norm Regularization.
- (Alexa) Revised code in long short term memory file and reran. Removed callbacks, as our models consistently performed better on training data without them. Very slight improvement in accuracy noted.

**Files Created:**
- work/13-neural-network-tuning.ipynb

**Impact:** Very minor improvement in accuracy noted (<0.1% difference). Best training accuracy remains at 7.0%.

**Next Steps:** Complete and run models using regularization techniques. Fix code in neural network tuning file and rerun. Transfer to Azure instance if necessary. Identify best neural network overall. Begin work on final project presentation and report.

## 2026-04-19 - Beginning the Process of the Final Report (All)
**Context:** Our work has begun to culminate into a final report summarizing our findings and bringing us closer to a conclusion.

**Work Completed:**
- (Ben) Created new folder finalreport within repository
- (Ben) Created new file, draft.md, in finalreport folder

**Files Created:**
- finalreport/draft.md

**Impact:** As we begin to conclude our work, the process has commenced on creating a final report in which we will summarize our findings and determine a final conclusion

**Next Steps:** Finish up any remaining work left and begin work on drafting our final report

## 2026-04-19 - Random Forest and XGBoost Full-Data Training Attempts (Shahaan)
**Context:** Attempting to run notebooks 09 and 11 on the full training dataset (113K rows) instead of the 10K subsample.

**Work Completed:**
- (Shahaan) Removed 10K subsample from notebooks 09 and 11; switched to full 113K training data for testing - Failed.
- (Shahaan) Fixed NaN F1 scores in RandomizedSearchCV by using `make_scorer` with `zero_division=0`
- (Shahaan) Added 30K-row search subsample to avoid RAM fragmentation during hyperparameter search
- (Shahaan) Added `USE_GPU` toggle to notebook 11 to switch between RTX 4060 and CPU

**Impact:** M5.T4, M5.T5 — Full-data RF and XGBoost training in progress; multiple memory fixes applied; pushing to Azure NC6s_v3 

**Next Steps:** Still working on getting notebooks 09 and 11 to complete successfully locally. Will push to Azure (NC6s_v3) for further testing.

## 2026-04-14 - Sequential Feature Engineering and Long Short Term Memory Model Creation (Alexa)
**Context:** Adding features based on order date by customer and creating a recurrent neural network using LSTM layers.

**Work Completed:**
- (Alexa) Added temporal features including purchase order grouped by customer and days since last purchase. Added Survey ResponseID back into the data and encoded using a label encoder.
- (Alexa) Re-ran our best performing neural network with the new data. Very slight improvement noted (training accuracy=12.0%, test accuracy=7.0%)
- (Alexa) Created a new neural network using an LSTM layer and reshaped the data appropriately to fit the model. 
- (Alexa) Added additional layers to LSTM network and tuned hyperparameters. Unable to improve accuracy using LSTM networks.

**Files Created:**
- work/12-lstm.ipynb

**Impact:** M3.T4 and M5.T6 completed. Best accuracy on the test data thus far is approximately 7.0%.

**Next Steps:** Identify best models and begin building presentation. Continue hyperparameter tuning of models in attempt to improve accuracy.

## 2026-04-12 - XGBoost Model (Shahaan)
**Context:** Building an XGBoost classifier for product category prediction (M5.T5)

**Work Completed:**
- (Shahaan) Created XGBoost notebook mirroring the Random Forest notebook structure for comparability
- (Shahaan) Built baseline and tuned models with feature importance, hyperparameter tuning (RandomizedSearchCV), and 5-fold cross-validation
- (Shahaan) Used `tree_method='hist'`, `n_jobs=1`, and 10K subsample to stay within memory constraints

**Files Created:**
- work/11-xgboost.ipynb

**Next Steps:** Run and evaluate on Azure; compare results against Random Forest and Wide & Deep NN benchmarks.

## 2026-04-10 - Wide & Deep Neural Network Hyperparameter Tuning (Alexa)
**Context:** Tuning hyperparameters of previously built neural networks. 

**Work Completed:**
- (Alexa) Added additional layers and neurons to wide & deep neural network
- (Alexa) Implemented validation_split hyperparameter in fit() methods to monitor and prevent overfitting
- (Alexa) Implemented EarlyStopping and ReduceLROnPlateau callbacks to wide & deep neural network
- (Alexa) Tested our best wide & deep neural network with different activation functions in the inner layers, including ELU, SELU, GELU, Swish, and Mish

**Impact:** M5.T7 completed. Tuned models did not improve results. Best accuracy of all of our neural networks is 6.9%.

**Next Steps:** Continue building neural networks. Build XGBoost model. Begin identifying the best models to present.

## 2026-04-05 - Wide & Deep Neural Network Creation (Alexa and Ben)
**Context:** Building MLP and Wide & Deep Neural Networks to classify the "Category" variable.

**Work Completed:**
- (Alexa) Built a simple MLP network for classification (training accuracy=12.6%, test accuracy=6.9%).
- (Alexa) Created a Wide & Deep Neural Network model, which resulted in very poor training and test accuracy (training accuracy=0.03%, test accuracy=0.03%).
- (Alexa) Removed some highly correlated features from the train and test sets to facilitate feature subsetting for wide & deep neural network. 
- (Alexa) Ran MLP network and wide & deep network on smaller dataset. Minimal improvement to accuracy for both models.
- (Ben) Subsetted and split features to train wide and deep components so they focus on different features.
- (Ben) Tested accuracy with split features  (training accuracy=0.9% - 1.0%, testing accuracy=6.4%).

**Files Created:**
- work/10-neural-networks.ipynb

**Next Steps:** Continue building neural networks, tune hyperparameters to improve neural networks (i.e., activation, epochs).

## 2026-03-29 - Random Forest Model (Shahaan)
**Context:** Building a Random Forest classifier for product category prediction (M5.T4)

**Work Completed:**
- (Shahaan) Created Random Forest notebook with baseline and tuned models on 10K subsample
- (Shahaan) Performed feature importance analysis and hyperparameter tuning (RandomizedSearchCV)
- (Shahaan) Evaluated with precision, recall, F1, confusion matrix, and 5-fold cross-validation
- (Shahaan) Memory constraints required subsampling, full dataset (113K rows) caused OOM errors

**Files Created:**
- work/09-random-forest.ipynb

**Next Steps:** Build XGBoost model (M5.T5)

## 2026-03-29 - Continued Association Rule Mining (Alexa)
**Context:** Continuing association rule mining and adding visualizations to identify itemsets with high support and lift.

**Work Completed:**
- (Alexa) Expanded association rule mining efforts to understand support and lift, in addition to confidence
- (Alexa) Identified association rules with greatest support to understand itemsets that frequently occur together
- (Alexa) Identified association rules with greatest lift to understand the relationships between antecedents and consequents (i.e., which antecedents increase the likelihood of which consequents the greatest)
- (Alexa) Created heatmaps to visualize the itemsets with the greatest confidence, support, and lift

**Impact:** Further association rule mining completed and results visualized to understand which itemsets have the greatest confidence, support, and lift

**Next Steps:** 

## 2026-03-29 - Began Bayesian rule mining (Shahaan)
**Context:** Attempting to add more data into dataset

**Work Completed:**
- (Shahaan) Met with Librarian on 3-24-2026 to find additional consumer data. Unsuccessful as she believes this data is good.

**Impact:** Looked for additional data with SU Librarian to no success 

## 2026-03-22 - Began Bayesian rule mining (Ben, Shahaan)
**Context:** Attempting Bayesian rule mining based off association rule mining as per professor suggestion

**Work Completed:**
- (Ben) - Read in copy of the data before encoding steps to facilitate Bayesian rule mining
- (Ben) - Began Bayesian rule mining attempts to identtify item categories that are commonly purchased by the same customer based on prior knowledge or probabilistic beliefs, essentially updating beliefs based on evidence
- (Ben) - Added comments to calculations made during preprocessing for Bayesian rule mining
- (Shahaan) Fixed data path from Codespaces to relative path for local development in vscode and created requirements.txt for virtual environment setup
- (Shahaan) Added Bayesian Network graph visualization using networkx
- (Shahaan) Added display of top Bayesian rules filtered by Kemeny-Oppenheim measure
- (Shahaan) Fit Bayesian Network parameters and displayed conditional probability tables (CPTs)
- (Shahaan) Added category connectivity bar chart showing in/out degree per category
- (Shahaan) Added Bayesian Network graph visualization using networkx

**Files Created**
- work/08-bayesian-rule-mining.ipynb

**Impact:** Identified category sets with high support and/or high confidence based on existing evidence found through association rule mining and probabilistic beliefs, ultimately updating beliefs on what categories of items are often purchased by the same customer and verifying or potentially disputing prior evidence regarding what to recommend customers based on what they purchase.

**Next Steps:** Continuing Bayesian rule mining efforts and examining additional measures as well as discussing possibility of obtaining an Azure instance during next team meeting.

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

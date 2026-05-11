Ok, nice job.  This is a tricky dataset to work with, and you've clearly put effort in.

Some notes as I went through:

- Figure 1: For heavily skewed distributions, log-scale the y-axis
- Figure 2: Bar chart does correspond to the text?
- "Bayesian Network" - [*Note the following was my initial reaction, before I read the text explaining the approach; the discussed approach addresses any concerns I had, but when presenting something like this, you might let the reader know that a fuller discussion of the BN is forthcoming*] I am not sure if this *is* a Bayesian Network - if it is, it *must* be a DAG (no cycles), edges imply causality, and you have sought to establish conditional probability tables.  With a little semantic fudging, you might say something like "A purchase in the home category causes a purchase in the battery category" but this is a stretch.  So, while visualization is useful in BNs, they are not used *for* visualization typically.
- Note the heatmap confusion matrix is nice, but because your color index is based on absolute, rather than normalized quantities, it gives you a biased view of your data - the "strongest diagonal" is strongest because those categories had the most purchases.

I think the big limitation here is that you didn't report baselines - majority class, stratified random.  You could have even done a first-order markov chain (a lot like your Bayesian Network).  I would have really liked to have seen how your numbers stack up against these baselines. The sharpest version of this here: your own feature importance analysis shows that `prev_parent_category` is the single most predictive feature by a wide margin (importance 0.1053).  So the obvious baseline is "predict next category = previous category" - a Markov-1 rule with no model at all.  If that gets, say, 30%, your 36.1% lifts only 6 points over a one-line rule.  If it gets 20%, your lift is 16 points and the modeling story is much stronger.  You can't know without running it, and given how dominant the feature is in your importance plot, this is the baseline that matters most.

Another thing - you should have evaluated your BN - it is, after all, a probabilistic model, and you could have even used its output along with other variables (demographics, etc.) to see if you could get an improvement.

I think you could have spent a little more time talking about the impact of class imbalance; this is clearly a predominating influence on your results.  Your best accuracy comes from those categories with the most instances.  I would have loved to have seen some additional discussion of this.  Related: the SMOTE result in notebook 16 - where the model collapsed to predicting only Home_Decor with weighted F1 of 0.0034 - is a striking negative result that you report but don't really dig into.  Why did SMOTE break this model so badly?  My guess is that SMOTE in a 156-dim space that's mostly one-hot doesn't generate sensible synthetic points, but a 30-minute diagnostic would have made the failure into a teachable result rather than a footnote.

One last thing: the 800-respondent subsample stood out to me as a self-imposed limit.  1M purchases × ~150 features should fit comfortably in memory on a typical machine, and you had Azure compute access for some of the RF/XGBoost work anyway.  Scaling to the full 5,027 respondents was technically straightforward, and would have given you better coverage of rare categories and demographic segments essentially for free.  You note it as a limitation, which is honest, but it's worth recognizing that this one was inside your control.

All that being said, this was a really strong project.  Nice work!

**Score: 28/30**


---

## Final Project Grade
| Assessment Item | Alexa Lotano | Shahaan Khan | Ben Euto |
|---|---|---|---|
| **Proposal (5 pts)** | 5 | 5 | 5 |
| **Midterm Report (10 pts)** | 10 | 10 | 10 |
| **Final Presentation (5 pts)** | 5 | 5 | 5 |
| **Final Report (30 pts)** | 28 | 28 | 28 |
| **Weekly Updates (30 pts)** | 30 | 30 | 28 |
| **Total (80 pts)** | **78** | **78** | **76** |

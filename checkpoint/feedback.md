- Does your proposal include all of the above mentioned sections? 1/1
- Are your objectives concrete and do you have a clear stakeholder need? 2/2
- Do you have a good data source and have you done a thorough job investigating its provenance and credibility? 1/1

Note - just because things come from Harvard Dataverse doesn't really provide any indication of data quality.  But you've done a nice job nonetheless here.

- Did you do a thorough job exploring your data? 2/2
- Have you done some initial modeling of your problem and do you have some early baseline results? 3/3

Ok, this is great.  It's not surprising to me that demographics are not in general highly predictive of purchasing in general.  One thing I wasn't sure of here was which categories you were targeting?  The more classes you have, the harder this is going to be.  And I'm not positive that PCA is the right way to go - you might instead try to whittle your features down a little.  I imagine may product categories will be time correlated, and others will not.  If you evaluate you predictions on a category by category basis (binary predictors) you'll have a much better sense of where the challenges lie.  One difficulty with your analysis right now is that you have a globally *bad* result, but no idea *why* it's bad.  So you want to try to start thinking more analytically - what modeling can you do that will reveal patterns in your data?  How can you use this to inform your modeling?

Thinking about this a little more - do you think *categories* are sufficient for purchasing rules, or recommendations, for that matter?  It's really going to depend on the category granularity I guess.  If I purchase a tent, maybe I'll purchase some tent stakes, and then shock cord, and then a backpack, and a headlamp.  But if all of these are in camping gear, that's not super useful.  In the end, categories might not provide enough of an informative signal anyway.

Association rule mining is a fast, unsupervised algorithm, and you could feasibly post-process that to drive a temporal algorithm - ARM is set based, not sequence based, but you could potentially use it to lift out positive examples and then train a model on just those.  You might also look into Bayesian rule mining.  There's been recent work in that space that could give you some new insights.

- Do you have a clear path forward? 1

I think you're doing fine here.  Start narrow, trying to find a few things you *can* predict successfully, and then expand from there.

Also - if you need more compute, I can get you an Azure instance with more horsepower.  Let me know.


Score: 10/10



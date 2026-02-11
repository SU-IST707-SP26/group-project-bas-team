Ok, this is promising.  I'm guessing the data is already available, and if this is crowdsourced (i.e., users voluntarily provided their data) then "ethics" are not a problem here.

What *is* a problem is scope.  This is incredibly broad, and you've thrown just about everything you can at it - it's rare that one would do both association rule mining and LSTMs.  And you're going to need to think about how far in advance you want your prediction window to be.  If someone orders toilet paper in bulk, chances are they're going to do it again at some quantifiable point in the future - but it might not be for 3 months (depending on how much TP they buy!).

So, I'd encourage your to think a little more narrowly about domain - types of products, re-purchases in the future vs. "you might also like:" type purchases.  And you might also think about more sophisticated models - you might have demographics in your data, but presumably wouldn't have this in a realistic context.  So, perhaps you can infer demographics based on past purchases?  This gets a little creepy, and yes, there is an ethical angle here.  But I think you can just be aware of this and think a little about how you would use and publish your data. 

But in general, I think the overall project is sound and look forward to your progress.

SCORE: 5/5
#sentiment analysis using Pattern library
from pattern.en import sentiment

#tested with amazon customer review
with open('5-star-review.txt') as f:
    text = f.read()

#output of senitment: (polarity, subjectivity)

#polarity - sentiment score between -1 and 1. 1 is most positive, -1 is most negative. 
polarity = sentiment(text)[0]
#subjectivity - value between 0 and 1. Higher subjectivity -> more personal (rather than factual info)
subjectivity = sentiment(text)[1]

#print statements to output overall sentiment and subjectivity
if polarity <= 0.05:
    polarity_pct = polarity*100
    print("Text was {}% negative.".format(polarity_pct))
    print("Overall, text was negative.")

elif polarity >= 0.05:
    polarity_pct = polarity*100
    print("Text was {}% positive.".format(polarity_pct))
    print("Overall, text was positive.")

else:
    print("Overall, text was neutral.")

if subjectivity <= 0.5:
    subjectivity_pct = subjectivity*100
    print("Text was {}% objective".format(subjectivity_pct))

else:
    subjectivity_pct = subjectivity*100
    print("Text was {}% subjective".format(subjectivity_pct))

#sentiment analysis using Vader library
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#function analyzing given text 
def sentiment_score(text):

    #creating analyzer object
    analyze_obj = SentimentIntensityAnalyzer()

    #storing polarity scores in dictionary variable
    sentiment_dict = analyze_obj.polarity_scores(text)

    #print statements to easily read output
    print("Text was rated ", sentiment_dict['neg']*100, "% Negative")
    print("Text was rated ", sentiment_dict['neu']*100, "% Neutral")
    print("Text was rated ", sentiment_dict['pos']*100, "% Positive")

    if sentiment_dict['compound'] >= 0.05 :
        print("Overall, text was positive")
 
    elif sentiment_dict['compound'] <= -0.05 :
        print("Overall, text was negative")
 
    else :
        print("Overall, text was neutral")

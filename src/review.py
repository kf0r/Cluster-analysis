from textblob import TextBlob
from datetime import datetime

class Review:
    '''
    Class representing a single review.
    Parameters:
        user_id (str): user id
        product_id (str): product id
        date (str): date of review
        score (float): score of review
        text (str): text of review
        sentiment (float): sentiment of review obtained from TextBlob
    '''
    def __init__(self, user_id, product_id, date, score, text):
        if not user_id or not isinstance(user_id, str):
            raise ValueError(f"Faulty user id {user_id}")
        if not product_id or not isinstance(product_id, str):
            raise ValueError(f"Faulty product id {product_id}")
        self.user_id = user_id
        self.product_id = product_id
        if not date:
            raise ValueError(f"Faulty date {date}")
        self.date = datetime.utcfromtimestamp(int(date))
        if not score:
            raise ValueError(f"Faulty score {score}")
        self.score = float(score)
        if not isinstance(text, str):
            raise ValueError(f"Faukty text {text}")
        self.sentiment = TextBlob(text).sentiment.polarity
        self.text = text

    def __repr__(self):
        return f"Review(user={self.user_id}, product={self.product_id}, date={self.date}, score={self.score}, sentiment={self.sentiment})"
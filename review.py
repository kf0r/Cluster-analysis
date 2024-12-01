from textblob import TextBlob
from datetime import datetime

class Review:
    def __init__(self, user_id, product_id, date, score, text):
        # Walidacja user_id i product_id
        if not user_id or not isinstance(user_id, str):
            raise ValueError(f"Nieprawidłowy identyfikator użytkownika: {user_id}")
        if not product_id or not isinstance(product_id, str):
            raise ValueError(f"Nieprawidłowy identyfikator produktu: {product_id}")
        self.user_id = user_id
        self.product_id = product_id

        # Walidacja daty
        if not date or not date.isdigit():
            raise ValueError(f"Nieprawidłowa wartość daty: {date}")
        self.date = datetime.utcfromtimestamp(int(date))

        # Walidacja oceny
        if not score or not score.replace('.', '', 1).isdigit():
            raise ValueError(f"Nieprawidłowa wartość oceny: {score}")
        self.score = float(score)

        # Walidacja tekstu recenzji
        if not text or not isinstance(text, str):
            raise ValueError(f"Nieprawidłowy tekst recenzji: {text}")
        self.sentiment = TextBlob(text).sentiment.polarity

    def __repr__(self):
        return f"Review(user={self.user_id}, product={self.product_id}, date={self.date}, score={self.score}, sentiment={self.sentiment})"
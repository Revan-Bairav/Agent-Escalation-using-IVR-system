import torch
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

def predict_sentiment(query, model, tokenizer, device):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    sentiment_score = probabilities[0][1].item()  # Probability of positive sentiment
    
    if sentiment_score > 0.6:
        return "Positive"
    elif sentiment_score < 0.4:
        return "Negative"
    else:
        return "Neutral"

def extract_keywords(query, num_keywords=5):
    tokens = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
    fdist = FreqDist(tokens)
    return [word for word, _ in fdist.most_common(num_keywords)]

def analyze_query(query, model, tokenizer, device):
    sentiment = predict_sentiment(query, model, tokenizer, device)
    keywords = extract_keywords(query)
    return {
        "query": query,
        "sentiment": sentiment,
        "keywords": keywords
    }
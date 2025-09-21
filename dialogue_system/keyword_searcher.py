import re
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(text): 
    text = text.lower() 
    text = re.sub(r'[^\w\s]', '', text) 
    tokens = text.split() 
    return tokens

def tfidf_ranking(utterance, domain_list):
    vectorizer = TfidfVectorizer()
    corpus = domain_list + [utterance]
    tfidf_matrix = vectorizer.fit_transform(corpus)

    sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    ranked = sorted(zip(domain_list, sims[0]), key=lambda x: x[1], reverse=True)
    return ranked

class RestaurantSearcher:
    def __init__(self, restaurant_manager):
        self.restaurant_manager = restaurant_manager
        self.unique_pricerange = None
        self.unique_area = None
        self.unique_food = None
        self.keywords = {
            "food": ["food", "restaurant", "serves"],
            "area": ["area", "part", "region", "side"],
            "pricerange": ["price", "pricerange", "cost"]
        }

    def search(self, utterance, attribute, window_size=2):
        if attribute not in ["pricerange", "area", "food"]:
            raise ValueError("Attribute must be one of 'pricerange', 'area', or 'food'.")

        domain_list = self.restaurant_manager.get_labels(attribute)
        tokens = preprocess(utterance)

        # --- Direct Match Check first ---
        for token in tokens:
            # Convert domain_list terms to lower for comparison
            lower_domain_list = [d.lower() for d in domain_list]
            if token.lower() in lower_domain_list:
                # Found a direct match, update and return
                if attribute == "food":
                    self.unique_food = token
                elif attribute == "area":
                    self.unique_area = token
                elif attribute == "pricerange":
                    self.unique_pricerange = token
                return token # Return the directly matched term

        context_words = set()
        for i, token in enumerate(tokens):
            if token in self.keywords[attribute]:
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(tokens))
                for j in range(start, end):
                    if i != j:
                        context_words.add(tokens[j])

        best_match = None
        min_distance = float("inf")

        for context_word in context_words:
            for term in domain_list:
                distance = Levenshtein.distance(context_word.lower(), term.lower())
                if distance < min_distance:
                    min_distance = distance
                    best_match = term

        if best_match and min_distance <= 3:
            if attribute == "food":
                self.unique_food = best_match
            elif attribute == "area":
                self.unique_area = best_match
            elif attribute == "pricerange":
                self.unique_pricerange = best_match
            return best_match

        ranked = tfidf_ranking(utterance.lower(), domain_list)
        if ranked:
            top_term, score = ranked[0]
            if score >= 0.5:
                if attribute == "food":
                    self.unique_food = top_term
                elif attribute == "area":
                    self.unique_area = top_term
                elif attribute == "pricerange":
                    self.unique_pricerange = top_term
                return top_term
            
        return None

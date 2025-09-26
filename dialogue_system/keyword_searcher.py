import re
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dialogue_system.types import SearchThemes

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
        self.unique_touristic = None
        self.unique_assigned_seats = None
        self.unique_children = None
        self.unique_romantic = None
        self.keywords = {
            SearchThemes.food: ["food", "restaurant", "serves"],
            SearchThemes.area: ["area", "part", "region", "side"],
            SearchThemes.pricerange: ["price", "pricerange", "cost"],
            SearchThemes.touristic: ["touristic", "tourist", "popular", "famous"],
            SearchThemes.assigned_seats: ["assigned seats", "seating", "seat", "assign"],
            SearchThemes.children: ["children", "kids", "child", "family"], 
            SearchThemes.romantic: ["romantic", "romance", "couple", "date"],
        }

    def search(self, utterance, attribute, window_size=2):
        if attribute not in [SearchThemes.pricerange, SearchThemes.area, SearchThemes.food, SearchThemes.touristic, SearchThemes.assigned_seats, SearchThemes.children, SearchThemes.romantic]:
            raise ValueError("Attribute must be one of 'pricerange', 'area', or 'food', 'touristic', 'assigned_seats', 'children', 'romantic'.")

        tokens = preprocess(utterance)

        domain_list = []

        if attribute in [SearchThemes.pricerange, SearchThemes.area, SearchThemes.food]:
            domain_list = self.restaurant_manager.get_labels(attribute.value)
        else:
            if attribute == SearchThemes.touristic:
                domain_list = SearchThemes.touristic.value.split()
            elif attribute == SearchThemes.assigned_seats:
                domain_list = SearchThemes.assigned_seats.value.split()
            elif attribute == SearchThemes.children:
                domain_list = SearchThemes.children.value.split()
            elif attribute == SearchThemes.romantic:
                domain_list = SearchThemes.romantic.value.split()

        for token in tokens:
            lower_domain_list = [d.lower() for d in domain_list]
            if token.lower() in lower_domain_list:
                if attribute == SearchThemes.food:
                    self.unique_food = token
                elif attribute == SearchThemes.area:
                    self.unique_area = token
                elif attribute == SearchThemes.pricerange:
                    self.unique_pricerange = token
                return token

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
            if attribute == SearchThemes.food:
                self.unique_food = best_match
            elif attribute == SearchThemes.area:
                self.unique_area = best_match
            elif attribute == SearchThemes.pricerange:
                self.unique_pricerange = best_match
            elif attribute == SearchThemes.touristic:
                self.unique_touristic = best_match
            elif attribute == SearchThemes.assigned_seats:
                self.unique_assigned_seats = best_match
            elif attribute == SearchThemes.children:
                self.unique_children = best_match
            elif attribute == SearchThemes.romantic:
                self.unique_romantic = best_match
            return best_match

        ranked = tfidf_ranking(utterance.lower(), domain_list)
        if ranked:
            top_term, score = ranked[0]
            if score >= 0.5:
                if attribute == SearchThemes.food:
                    self.unique_food = top_term
                elif attribute == SearchThemes.area:
                    self.unique_area = top_term
                elif attribute == SearchThemes.pricerange:
                    self.unique_pricerange = top_term
                elif attribute == SearchThemes.touristic:
                    self.unique_touristic = top_term
                elif attribute == SearchThemes.assigned_seats:
                    self.unique_assigned_seats = top_term
                elif attribute == SearchThemes.children:
                    self.unique_children = top_term
                elif attribute == SearchThemes.romantic:
                    self.unique_romantic = top_term
                return top_term
            
        return None

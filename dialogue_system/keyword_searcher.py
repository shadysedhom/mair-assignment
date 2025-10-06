import re
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dialogue_system.types import SearchThemes


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


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
            SearchThemes.area: ["area", "part", "region", "side", "in", "at"],
            SearchThemes.pricerange: ["price", "pricerange", "cost", "expensive", "cheap"],
            SearchThemes.touristic: ["touristic", "tourist", "popular", "famous"],
            SearchThemes.assigned_seats: ["assigned", "seats", "seating", "seat"],
            SearchThemes.children: ["children", "kids", "child", "family"],
            SearchThemes.romantic: ["romantic", "romance", "couple", "date"],
        }

    def _contextual_search(self, tokens, domain_list, attribute, window_size=2):
        """Look around relevant keywords and use Levenshtein distance to find best match."""
        context_words = set()
        for i, token in enumerate(tokens):
            if token in self.keywords[attribute]:
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(tokens))
                for j in range(start, end):
                    if i != j:
                        context_words.add(tokens[j])

        best_match, min_distance = None, float("inf")

        for context_word in context_words:
            for term in domain_list:
                distance = Levenshtein.distance(context_word.lower(), term.lower())
                if distance <= min_distance:
                    min_distance = distance
                    best_match = term

        if best_match and min_distance <= 3:
            return best_match
        return None

    def _global_search(self, utterance, domain_list):
        """Use Levenshtein + TF-IDF across the full utterance."""
        tokens = preprocess(utterance)
        best_match, min_distance = None, float("inf")
        for token in tokens:
            for term in domain_list:
                distance = Levenshtein.distance(token.lower(), term.lower())
                if distance <= min_distance:
                    min_distance = distance
                    best_match = term
        if best_match and min_distance <= 3:
            return best_match

        ranked = tfidf_ranking(utterance.lower(), domain_list)
        if ranked:
            top_term, score = ranked[0]
            if score >= 0.5:
                return top_term
        return None

    def search(self, utterance, attribute, window_size=2):
        """Main entry point for searching attributes in user utterance."""
        if attribute not in [
            SearchThemes.pricerange,
            SearchThemes.area,
            SearchThemes.food,
            SearchThemes.touristic,
            SearchThemes.assigned_seats,
            SearchThemes.children,
            SearchThemes.romantic,
        ]:
            raise ValueError("Invalid attribute for restaurant search.")

        stopwords = {
            "the", "a", "an", "in", "on", "at", "for", "to", "of", "is", "it",
            "let", "lets", "do", "does", "be", "am", "are", "was", "were",
            "please", "this", "that", "those", "these", "with", "and", "but", "or"
        }

        tokens = [t for t in preprocess(utterance) if t not in stopwords]
        cleaned_utterance = " ".join(tokens)

        if attribute in [SearchThemes.pricerange, SearchThemes.area, SearchThemes.food]:
            domain_list = self.restaurant_manager.get_labels(attribute.value)
        else:
            domain_list = SearchThemes[attribute.name].value.split()

        result = self._contextual_search(tokens, domain_list, attribute, window_size)

        if not result:
            result = self._global_search(cleaned_utterance, domain_list)

        if result:
            current_value = getattr(self, f"unique_{attribute.value}", None)
            if current_value is None:
                setattr(self, f"unique_{attribute.value}", result)
            return result

        return None

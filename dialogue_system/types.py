from enum import Enum

class InferenceTypes(Enum):
    cheap = "cheap"
    good_food = "good food"
    romanian = "romanian"
    busy = "busy"
    long_stay = "long stay"

class ConsequenceTypes(Enum):
    touristic = "touristic"
    assigned_seats = "assigned seats"
    children = "children"
    romantic = "romantic"

class SearchThemes(Enum):
    food = "food"
    area = "area"
    pricerange = "pricerange"
    touristic = "touristic"
    assigned_seats = "assigned seats"
    children = "children"
    romantic = "romantic"
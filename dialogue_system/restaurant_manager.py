from __future__ import annotations
from dialogue_system.restaurant import Restaurant
from dialogue_system.types import SearchThemes


class RestaurantManager:
    def __init__(self, restaurants):
        self.restaurants = restaurants
        self.unique_priceranges = self._get_unique(SearchThemes.pricerange.value)
        self.unique_areas = self._get_unique(SearchThemes.area.value)
        self.unique_foods = self._get_unique(SearchThemes.food.value)

    def _get_unique(self, attribute):
        """Helper method to get unique values for a given Restaurant attribute"""
        return sorted(list({getattr(r, attribute) for r in self.restaurants}))
    
    def get_labels(self,label):
        if label == SearchThemes.pricerange.value:
            return self.unique_priceranges
        elif label == SearchThemes.area.value:
            return self.unique_areas
        elif label == SearchThemes.food.value:
            return self.unique_foods
        else:
            raise ValueError("Label must be one of 'pricerange', 'area', or 'food'.")

    def find_restaurants(self, area: str = None, pricerange: str = None, food: str = None) -> list[Restaurant]:
        """
        Filters the list of restaurants based on specified criteria.

        Args:
            area: The desired area.
            pricerange: The desired price range.
            food: The desired food type.

        Returns:
            A list of Restaurant objects that match all specified criteria.
            Returns an empty list if no matches are found.
        """
        
        potential_matches = self.restaurants

        # A 'don't care' preference can be represented by the value "any" or by the argument being None.
        # We will filter step-by-step.
        if area and area != "any":
            potential_matches = [r for r in potential_matches if r.area.lower() == area.lower()]

        if pricerange and pricerange != "any":
            potential_matches = [r for r in potential_matches if r.pricerange.lower() == pricerange.lower()]

        if food and food != "any":
            potential_matches = [r for r in potential_matches if r.food.lower() == food.lower()]

        return potential_matches
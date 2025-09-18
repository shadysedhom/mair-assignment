class RestaurantManager:
    def __init__(self, restaurants):
        self.restaurants = restaurants
        self.unique_priceranges = self._get_unique("pricerange")
        self.unique_areas = self._get_unique("area")
        self.unique_foods = self._get_unique("food")

    def _get_unique(self, attribute):
        """Helper method to get unique values for a given Restaurant attribute"""
        return sorted(list({getattr(r, attribute) for r in self.restaurants}))
    
    def get_labels(self,label):
        if label == "pricerange":
            return self.unique_priceranges
        elif label == "area":
            return self.unique_areas
        elif label == "food":
            return self.unique_foods
        else:
            raise ValueError("Label must be one of 'pricerange', 'area', or 'food'.")

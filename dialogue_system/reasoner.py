from typing import Dict, Tuple
from dialogue_system.types import InferenceTypes, ConsequenceTypes
from dialogue_system.restaurant import Restaurant 


class Inference:
    def __init__(self, can_be_true: bool = False, can_be_false: bool = False):
        self.can_be_true = can_be_true
        self.can_be_false = can_be_false

    def add_true(self):
        self.can_be_true = True

    def add_false(self):
        self.can_be_false = True

    def as_tuple(self) -> Tuple[bool, bool]:
        return (self.can_be_true, self.can_be_false)

from typing import Dict, Tuple


class Inference:
    def __init__(self, can_be_true: bool = False, can_be_false: bool = False):
        self.can_be_true = can_be_true
        self.can_be_false = can_be_false

    def add_true(self):
        self.can_be_true = True

    def add_false(self):
        self.can_be_false = True

    def as_tuple(self) -> Tuple[bool, bool]:
        return (self.can_be_true, self.can_be_false)

    def __repr__(self):
        return f"Inference(true={self.can_be_true}, false={self.can_be_false})"


def apply_inference(restaurant: Restaurant, touristic=None, assigned_seats=None, children=None, romantic=None):
    inferred: Dict[ConsequenceTypes, Inference] = {}
    contradictions = {}

    if touristic is not None:
        if restaurant.pricerange == InferenceTypes.cheap.value and restaurant.food_quality == InferenceTypes.good_food.value or restaurant.food_quality == "good":
            if ConsequenceTypes.touristic in inferred and inferred[ConsequenceTypes.touristic].can_be_false:
                contradictions[ConsequenceTypes.touristic] = "Touristic status contradicts cheap and good food rule."
            inferred.setdefault(ConsequenceTypes.touristic, Inference()).add_true()

        if restaurant.food == InferenceTypes.romanian.value:
            if ConsequenceTypes.touristic in inferred and inferred[ConsequenceTypes.touristic].can_be_true:
                contradictions[ConsequenceTypes.touristic] = "Touristic status contradicts Romanian cuisine rule."
            inferred.setdefault(ConsequenceTypes.touristic, Inference()).add_false()

    if assigned_seats is not None:
        if restaurant.crowdedness == InferenceTypes.busy.value:
            inferred.setdefault(ConsequenceTypes.assigned_seats, Inference()).add_true()

    if children is not None:
        if restaurant.length_of_stay == InferenceTypes.long_stay.value or restaurant.length_of_stay == "long":
            inferred.setdefault(ConsequenceTypes.children, Inference()).add_false()

    if romantic is not None:
        if restaurant.crowdedness == InferenceTypes.busy.value:
            if ConsequenceTypes.romantic in inferred and inferred[ConsequenceTypes.romantic].can_be_true:
                contradictions[ConsequenceTypes.romantic] = "Romantic status contradicts busy rule."
            inferred.setdefault(ConsequenceTypes.romantic, Inference()).add_false()

        if restaurant.length_of_stay == InferenceTypes.long_stay.value or restaurant.length_of_stay == "long":
            if ConsequenceTypes.romantic in inferred and inferred[ConsequenceTypes.romantic].can_be_false:
                contradictions[ConsequenceTypes.romantic] = "Romantic status contradicts long stay rule."
            inferred.setdefault(ConsequenceTypes.romantic, Inference()).add_true()

    return inferred, contradictions



def reason_about_restaurants(
    candidates: list[Restaurant],
    touristic=None,
    assigned_seats=None,
    children=None,
    romantic=None,
    max_recommendations=None
):
    possible_recommendations = []
    for restaurant in candidates:
        inferred, contradictions = apply_inference(restaurant, touristic, assigned_seats, children, romantic)
        reasoning = []
        is_recommended = True

        if ConsequenceTypes.touristic in inferred:
            inf = inferred[ConsequenceTypes.touristic]
            if inf.can_be_true:
                reasoning.append("It is cheap and has good food, so it attracts tourists.")
            if inf.can_be_false:
                reasoning.append("The food is Romanian, so it is not touristic.")
                is_recommended = False

        if ConsequenceTypes.assigned_seats in inferred:
            inf = inferred[ConsequenceTypes.assigned_seats]
            if inf.can_be_true:
                reasoning.append("It is usually busy, so the waiter assigns seats.")

        if ConsequenceTypes.romantic in inferred:
            inf = inferred[ConsequenceTypes.romantic]
            if inf.can_be_true:
                reasoning.append("Guests tend to stay long, which makes it romantic.")
            if inf.can_be_false:
                reasoning.append("It is busy, so it is not romantic.")
                is_recommended = False

        if ConsequenceTypes.children in inferred:
            inf = inferred[ConsequenceTypes.children]
            if inf.can_be_false:
                reasoning.append("Long stays make it less suitable for children.")
                is_recommended = False

        if contradictions:
            reasoning.append("Contradictions found: " + "; ".join(contradictions.values()))
            print(f"Not recommended: {restaurant.name}")
            is_recommended = False
        elif not is_recommended:
            print(f"Not recommended: {restaurant.name}")
        else:
            if inferred:
                print(f"Recommended: {restaurant.name}")
            else:
                print(f"Recommended (no inferences): {restaurant.name}")

        print("Reasoning:")
        for r in reasoning:
            print(f"- {r}")

        if max_recommendations is None or (len(possible_recommendations) < max_recommendations and is_recommended):
            if is_recommended:
                possible_recommendations.append(restaurant)

        print("\n" + "-"*40 + "\n")

    return possible_recommendations



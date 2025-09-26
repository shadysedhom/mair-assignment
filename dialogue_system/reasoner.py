from dialogue_system.types import InferenceTypes, ConsequenceTypes
from dialogue_system.restaurant import Restaurant 


def apply_inference(restaurant: Restaurant, touristic=None, assigned_seats=None, children=None, romantic=None):
    inferred = {}
    contradictions = {}

    if touristic is not None:
        if restaurant.pricerange == InferenceTypes.cheap.value and restaurant.food_quality == InferenceTypes.good_food.value:
            if ConsequenceTypes.touristic in inferred and inferred[ConsequenceTypes.touristic] == False:
                contradictions[ConsequenceTypes.touristic] = "Touristic status contradicts cheap and good food rule."
            inferred[ConsequenceTypes.touristic] = True

        if restaurant.food == InferenceTypes.romanian.value:
            if ConsequenceTypes.touristic in inferred and inferred[ConsequenceTypes.touristic] == True:
                contradictions[ConsequenceTypes.touristic] = "Touristic status contradicts Romanian cuisine rule."
            inferred[ConsequenceTypes.touristic] = False
    if assigned_seats is not None:
        if restaurant.crowdedness == InferenceTypes.busy.value:
            inferred[ConsequenceTypes.assigned_seats] = True

    if children is not None:
        if restaurant.length_of_stay == InferenceTypes.long_stay.value or restaurant.length_of_stay == "long":
            inferred[ConsequenceTypes.children] = False

    if romantic is not None:
        if restaurant.crowdedness == InferenceTypes.busy.value:
            if ConsequenceTypes.romantic in inferred and inferred[ConsequenceTypes.romantic] == True:
                contradictions[ConsequenceTypes.romantic] = "Romantic status contradicts busy rule."
            inferred[ConsequenceTypes.romantic] = False

        if restaurant.length_of_stay == InferenceTypes.long_stay.value or restaurant.length_of_stay == "long":

            if ConsequenceTypes.romantic in inferred and inferred[ConsequenceTypes.romantic] == False:
                contradictions[ConsequenceTypes.romantic] = "Romantic status contradicts long stay rule."
            inferred[ConsequenceTypes.romantic] = True

    return inferred, contradictions


def reason_about_restaurants(
    candidates: list[Restaurant],
    touristic=None,
    assigned_seats=None,
    children=None,
    romantic=None
):
    possible_recommendations = []
    for restaurant in candidates:
        inferred, contradictions = apply_inference(restaurant, touristic, assigned_seats, children, romantic)
        reasoning = []

        is_recommended = True

        if inferred.get(ConsequenceTypes.touristic) == True:
            reasoning.append("It is cheap and has good food, so it attracts tourists.")
        elif inferred.get(ConsequenceTypes.touristic) == False:
            reasoning.append("The food is Romanian, so it is not touristic.")
            is_recommended = False
        if inferred.get(ConsequenceTypes.assigned_seats) == True:
            reasoning.append("It is usually busy, so the waiter assigns seats.")

        if inferred.get(ConsequenceTypes.romantic) == True:
            reasoning.append("Guests tend to stay long, which makes it romantic.")
        elif inferred.get(ConsequenceTypes.romantic) == False:
            reasoning.append("It is busy, so it is not romantic.")
            is_recommended = False

        if inferred.get(ConsequenceTypes.children) == False:
            reasoning.append("Long stays make it less suitable for children.")
            is_recommended = False

        if contradictions:
            reasoning.append("Contradictions found: " + "; ".join(contradictions.values()))
            print(f"Not recommended: {restaurant.name}")
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
            possible_recommendations.append(restaurant)
        print("\n" + "-"*40 + "\n")
    return possible_recommendations


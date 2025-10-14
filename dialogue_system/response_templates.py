import random

# A list of more human-like hints for food preferences
humanlike_food_hints = [
    "We have a lot of options! For example, you could go for something like Italian, Chinese, or maybe even Indian?",
    "Let's see... how about French, Thai, or Vietnamese? Any of those sound good?",
    "If you're feeling adventurous, we have British food, seafood, or some nice gastropubs."
]

# A list of more system-like hints for food preferences
system_food_hints = [
    "Supported food types include: Italian, Chinese, Indian, et cetera.",
    "Supported food types include: French, Thai, Vietnamese, et cetera.",
    "Supported food types include: British, Seafood, Gastropub, et cetera."
]

HUMANLIKE_TEMPLATES = {
    "welcome": "Hello and welcome to the restaurant recommender! I'd be happy to help you find the perfect place to eat. To get started, could you tell me what you have in mind? I'll need to know the area, preferred food, and a price range.",
    "ask_area": "Of course! What part of town are you thinking of?",
    "ask_area_invalid": "I'm sorry, I don't seem to recognize that area. Could you choose from one of these: {hint_options}?",
    "ask_food": "What type of food are you in the mood for?",
    "ask_food_invalid": lambda: random.choice(humanlike_food_hints),
    "ask_pricerange": "How much are you looking to spend? For example, cheap, moderate, or expensive?",
    "ask_pricerange_invalid": "My apologies, I don't have that price range in my system. The available options are: {hint_options}.",
    "no_results": "I'm sorry, it looks like I couldn't find any restaurants that match your request. Maybe we could try a different combination?",
    "suggest_restaurant": "I've found a place you might like! {name} is a lovely {food} restaurant in the {area} part of town. It's in the {pricerange} price range.",
    "ask_conformation": "Does this suggestion sound good to you?",
    "ask_part_incorrect": "Oh, I see. Could you tell me which part of my suggestion was incorrect so I can try again? Was it the area, the food, the price, or all of them?",
    "ask_preference_again": "No problem, let's start over. Could you tell me your preferences for area, food, and price range again?",
    "bye": "It was a pleasure helping you. Goodbye and enjoy your meal!",
    "show_possible_restaurants_count": "Okay, I found {count} restaurants that match what you're looking for:",
    "show_restaurant_details": "- There's {name}, which serves {food} food in the {pricerange} price range in the {area} area.",
    "ask_extra_preference": "Before I make a final suggestion, do you have any other requirements? For example, are you looking for a place that is touristic, romantic, good for children, or has assigned seating?",
    "confirm_term": "Just to be sure, did you mean '{term}' for {attribute}?"
}

SYSTEM_TEMPLATES = {
    "welcome": "Restaurant recommendation system activated. Specify preferences for area, food, and price range.",
    "ask_area": "Specify area.",
    "ask_area_invalid": "Error: Area not recognized. Valid options are: {hint_options}.",
    "ask_food": "Specify food type.",
    "ask_food_invalid": lambda: random.choice(system_food_hints),
    "ask_pricerange": "Specify price range (cheap, moderate, expensive).",
    "ask_pricerange_invalid": "Error: Price range not recognized. Valid options are: {hint_options}.",
    "no_results": "Query returned zero results.",
    "suggest_restaurant": "Recommendation: {name}. Area: {area}. Food: {food}. Price: {pricerange}.",
    "ask_conformation": "Confirm suggestion? (yes/no)",
    "ask_part_incorrect": "Specify incorrect attribute: Area, Food, Price Range, All.",
    "ask_preference_again": "Restarting preference elicitation. Specify area, food, and price range.",
    "bye": "Session terminated.",
    "show_possible_restaurants_count": "Query returned {count} results:",
    "show_restaurant_details": "- Restaurant: {name}. Attributes: food={food}, pricerange={pricerange}, area={area}.",
    "ask_extra_preference": "Specify additional preferences from the available options: touristic, assigned seats, romantic, children.",
    "confirm_term": "Confirm {attribute}: '{term}'? (yes/no)"
}

import random
from dialogue_system.finite_state_machine import FSM, State, Transition, Context, Inform, Affirm, Deny, Hello, Null,Negate
from dialogue_system import keyword_searcher
from dialogue_system.restaurant_manager import RestaurantManager


food_preference_hints = [
    "Some popular options are 'italian', 'chinese', or 'indian'.",
    "For example, you could try 'french', 'thai', or 'vietnamese'.",
    "You could also choose something like 'british', 'seafood', or 'gastropub'."
]

def initialize_fsm(keyword_searcher: keyword_searcher, ML_model, restaurant_manager: RestaurantManager) -> FSM:

    def _process_preferences(fsm: FSM, text_input: str):
        """Helper to extract all preferences from a user utterance and update the context."""
        area_output = fsm.keyword_searcher.search(text_input, "area")
        food_output = fsm.keyword_searcher.search(text_input, "food")
        pricerange_output = fsm.keyword_searcher.search(text_input, "pricerange")

        if area_output:
            fsm.context.area_known = True
            fsm.context.area = area_output
        if food_output:
            fsm.context.food_known = True
            fsm.context.food = food_output
        if pricerange_output:
            fsm.context.pricerange_known = True
            fsm.context.pricerange = pricerange_output
        
        return area_output, food_output, pricerange_output

    def welcome_action(fsm: FSM):
        print("Welcome! Let's start. What kind of restaurant are you looking for? Please inform me about your preferences (area, food, price range).")
        text = input("You: ")
        area_found, food_found, pricerange_found = _process_preferences(fsm, text)
        return fsm.ML_model.predict([text])[0]

    def ask_area_action(fsm: FSM):
        valid_options = fsm.restaurant_manager.get_labels('area')
        print("Which area would you like?")
        text_input = input("You: ")
        area_found, food_found, pricerange_found = _process_preferences(fsm, text_input)

        if not area_found:
            hint_options = ', '.join([opt for opt in valid_options if opt])
            print(f"I'm sorry, I don't recognize that area. Please choose from: {hint_options}.")

        return fsm.ML_model.predict([text_input])[0]

    def ask_food_action(fsm: FSM): 
        print("What type of food do you prefer?")
        text_input = input("You: ")
        area_found, food_found, pricerange_found = _process_preferences(fsm, text_input)

        if not food_found:
            print(random.choice(food_preference_hints))

        return fsm.ML_model.predict([text_input])[0]

    def ask_pricerange_action(fsm: FSM):
        valid_options = fsm.restaurant_manager.get_labels('pricerange')
        print("What price range are you looking for?")
        print()
        text_input = input("You: ")
        area_found, food_found, pricerange_found = _process_preferences(fsm, text_input)

        if not pricerange_found:
            hint_options = ', '.join([opt for opt in valid_options if opt])
            print(f"I'm sorry, I don't recognize that price range. Please choose from: {hint_options}.")

        return fsm.ML_model.predict([text_input])[0]

    def suggest_restaurant_action(fsm: FSM):
        matches = fsm.restaurant_manager.find_restaurants(
            area=fsm.context.area,
            pricerange=fsm.context.pricerange,
            food=fsm.context.food
        )

        if not matches:
            print("I'm sorry, there are no restaurants that match your request.")
            return "none" 

        suggestion = random.choice(matches)
        fsm.context.remaining_matches = [r for r in matches if r != suggestion]

        print(f"{suggestion.name} is a nice place in the {suggestion.area} part of town serving {suggestion.food} food in the {suggestion.pricerange} price range.")
        return "inform"
    
    def ask_conformation_action(fsm: FSM): 
        print("Is this suggestion okay for you?")
        text_input = input("You: ")
        action = fsm.ML_model.predict([text_input])[0]
        return action
    
    def ask_part_incorrect_action(fsm: FSM): 
        print("Which part of the suggestion was incorrect? (Area, Food, Price Range, All)")

        text_input = input("You: ")
        action = fsm.ML_model.predict([text_input])[0]

        if "area" in text_input.lower():
            fsm.context.incorrect_part = "area"
        elif "food" in text_input.lower():
            fsm.context.incorrect_part = "food"
        elif "price" in text_input.lower() or "pricerange" in text_input.lower():
            fsm.context.incorrect_part = "pricerange"
        elif "all" in text_input.lower():
            fsm.context.incorrect_part = "all"
        return action
    
    def ask_preference_action(fsm: FSM): 

        fsm.context.area_known = False
        fsm.context.food_known = False
        fsm.context.pricerange_known = False
        fsm.context.area = None
        fsm.context.food = None
        fsm.context.pricerange = None
        fsm.context.incorrect_part = None
    
        print("Please express your preferences again (area, food, price range).")
        text_input = input("You: ")
        area_found, food_found, pricerange_found = _process_preferences(fsm, text_input)
        return fsm.ML_model.predict([text_input])[0]

    def bye_action(fsm: FSM): 
        print("Goodbye!")
        fsm.is_active = False
        return "bye"

    welcome = State("welcome", welcome_action)
    ask_area = State("ask_area", ask_area_action)
    ask_food = State("ask_food", ask_food_action)
    ask_pricerange = State("ask_pricerange", ask_pricerange_action)
    suggest_restaurant = State("suggest_restaurant", suggest_restaurant_action)
    ask_conformation = State("ask_conformation", ask_conformation_action)
    ask_part_incorrect = State("ask_part_incorrect", ask_part_incorrect_action)
    ask_preference = State("ask_to_express_preference", ask_preference_action)
    bye = State("bye", bye_action)

    # Reordered transitions for welcome state
    welcome.add_transition(Transition(suggest_restaurant, lambda a, c: isinstance(a, (Inform, Hello, Null)) and c.area_known and c.food_known and c.pricerange_known))
    welcome.add_transition(Transition(ask_food, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.food_known))
    welcome.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.pricerange_known))
    welcome.add_transition(Transition(ask_area, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.area_known))

    # Reordered transitions for ask_area state
    ask_area.add_transition(Transition(suggest_restaurant, lambda a, c: isinstance(a, Inform) and c.area_known and c.food_known and c.pricerange_known))
    ask_area.add_transition(Transition(ask_food, lambda a, c: isinstance(a, Inform) and c.area_known and not c.food_known))
    ask_area.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, Inform) and c.area_known and not c.pricerange_known))
    ask_area.add_transition(Transition(ask_area, lambda a, c: isinstance(a, Inform) and not c.area_known))

    # Reordered transitions for ask_food state
    ask_food.add_transition(Transition(suggest_restaurant, lambda a, c: isinstance(a, Inform) and c.area_known and c.food_known and c.pricerange_known))
    ask_food.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, Inform) and c.food_known and not c.pricerange_known))
    ask_food.add_transition(Transition(ask_area, lambda a, c: isinstance(a, Inform) and c.food_known and not c.area_known))
    ask_food.add_transition(Transition(ask_food, lambda a, c: isinstance(a, Inform) and not c.food_known))

    # Reordered transitions for ask_pricerange state
    ask_pricerange.add_transition(Transition(suggest_restaurant, lambda a, c: isinstance(a, Inform) and c.area_known and c.food_known and c.pricerange_known))
    ask_pricerange.add_transition(Transition(ask_food, lambda a, c: isinstance(a, Inform) and c.pricerange_known and not c.food_known))
    ask_pricerange.add_transition(Transition(ask_area, lambda a, c: isinstance(a, Inform) and c.pricerange_known and not c.area_known))
    ask_pricerange.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, Inform) and not c.pricerange_known))

    suggest_restaurant.add_transition(Transition(ask_conformation, lambda a, c: isinstance(a, Inform)))
    suggest_restaurant.add_transition(Transition(ask_preference, lambda a, c: isinstance(a, Null)))
    ask_conformation.add_transition(Transition(bye, lambda a, c: isinstance(a, Affirm)))
    ask_conformation.add_transition(Transition(ask_part_incorrect, lambda a, c: isinstance(a, (Deny, Negate))))
    ask_conformation.add_transition(Transition(ask_conformation, lambda a, c: not isinstance(a, (Affirm, Deny))))

    ask_part_incorrect.add_transition(Transition(ask_preference, lambda a, c: isinstance(a, Inform) and c.incorrect_part == "all"))
    ask_part_incorrect.add_transition(Transition(ask_area, lambda a, c: isinstance(a, Inform) and c.incorrect_part == "area"))
    ask_part_incorrect.add_transition(Transition(ask_food, lambda a, c: isinstance(a, Inform) and c.incorrect_part == "food"))
    ask_part_incorrect.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, Inform) and c.incorrect_part == "pricerange"))

    ask_preference.add_transition(Transition(suggest_restaurant, lambda a, c: isinstance(a, (Inform, Hello, Null)) and c.area_known and c.food_known and c.pricerange_known))
    ask_preference.add_transition(Transition(ask_food, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.food_known))
    ask_preference.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.pricerange_known))
    ask_preference.add_transition(Transition(ask_area, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.area_known))


    ctx = Context()
    fsm = FSM(welcome, ctx, keyword_searcher, ML_model, restaurant_manager)

    return fsm

from Transition_states import FSM, State, Transition, Context, Action, Inform, Affirm, Deny, Hello, Bye, NoneAct
from dialogue_system import keyword_searcher


def initialize_fsm(keyword_searcher: keyword_searcher, ML_model) -> FSM:

    def welcome_action(fsm: FSM):
        print("Welcome! Let's start. What kind of restaurant are you looking for? Please inform me about your preferences (area, food, price range).")
        text = input("You: ")
        action = fsm.ML_model.predict([text])[0]
        area_output = fsm.keyword_searcher.search(text, "area")
        food_output = fsm.keyword_searcher.search(text, "food")
        pricerange_output = fsm.keyword_searcher.search(text, "pricerange")
        print(f"[DEBUG] Detected action: {action}, area: {area_output}, food: {food_output}, pricerange: {pricerange_output}")
        if area_output:
            fsm.context.area_known = True
            fsm.context.area = area_output
        if food_output:
            fsm.context.food_known = True
            fsm.context.food = food_output
        if pricerange_output:
            fsm.context.pricerange_known = True
            fsm.context.pricerange = pricerange_output
        return action

    def ask_area_action(fsm: FSM):
        print("Which area would you like?")
        text_input = input("You: ")
        action = fsm.ML_model.predict([text_input])[0]

        area_output = fsm.keyword_searcher.search(text_input, "area")
        if area_output:
            fsm.context.area_known = True
            fsm.context.area = area_output

        return action

    def ask_food_action(fsm: FSM): 
        print("What type of food do you prefer?")

        text_input = input("You: ")
        action = fsm.ML_model.predict([text_input])[0]

        food_output = fsm.keyword_searcher.search(text_input, "food")
        if food_output:
            fsm.context.food_known = True
            fsm.context.food = food_output
        return action

    def ask_pricerange_action(fsm: FSM):
        print("What price range are you looking for?")
        text_input = input("You: ")
        action = fsm.ML_model.predict([text_input])[0]

        pricerange_output = fsm.keyword_searcher.search(text_input, "pricerange")
        if pricerange_output:
            fsm.context.pricerange_known = True
            fsm.context.pricerange = pricerange_output
        return action

    def suggest_restaurant_action(fsm: FSM): 
        print("I suggest a restaurant for you.")
        print(f"Based on your preferences: Area - {fsm.context.area}, Food - {fsm.context.food}, Price Range - {fsm.context.pricerange}.")
        return "inform"

    def bye_action(fsm: FSM): 
        print("Goodbye!")
        return "bye"

    welcome = State("welcome", welcome_action)
    ask_area = State("ask_area", ask_area_action)
    ask_food = State("ask_food", ask_food_action)
    ask_pricerange = State("ask_pricerange", ask_pricerange_action)
    suggest_restaurant = State("suggest_restaurant", suggest_restaurant_action)
    bye = State("bye", bye_action)

    welcome.add_transition(Transition(suggest_restaurant, lambda a, c: isinstance(a, (Inform, Hello, NoneAct)) and c.area_known and c.food_known and c.pricerange_known))
    suggest_restaurant.add_transition(Transition(bye, lambda a, c: isinstance(a, Affirm) and c.area_known and c.food_known and not c.pricerange_known))
    welcome.add_transition(Transition(ask_area, lambda a, c: isinstance(a, (Inform, Hello, NoneAct)) and not c.area_known))
    welcome.add_transition(Transition(ask_food, lambda a, c: isinstance(a, (Inform, Hello, NoneAct)) and c.area_known and not c.food_known))
    welcome.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, (Inform, Hello, NoneAct)) and c.area_known and c.food_known and not c.pricerange_known))
    
    ctx = Context()
    fsm = FSM(welcome, ctx, keyword_searcher, ML_model)

    return fsm


# A dictionary to hold the templates for system utterances.
system_utterances = {
    "welcome": "Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range or food type. How may I help you?",
    "ask_area": "What part of town are you looking for?",
    "ask_pricerange": "And what price range are you looking for? (cheap, moderate, or expensive)",
    "ask_food": "What type of food would you like?",
    "no_results": "I'm sorry, there are no restaurants that match your request.",
    "offer_restaurant": "{restaurant_name} is a nice place in the {area} part of town serving {food} food in the {pricerange} price range.",
    "ask_more_info": "Would you like more information or another suggestion?",
    "goodbye": "Goodbye!"
}

from dialogue_system.finite_state_machine_initializor import initialize_fsm

def start_dialogue_system(model, restaurant_manager, restaurant_searcher, use_asr=False, use_tts=False, confirm_matches=False):
    """
    Launches the interactive restaurant dialogue system.
    """
    print("\n" + "-"*100)
    print("Welcome to the Restaurant Dialogue System!".center(100) + "\n" + "-"*100)
    
    fsm = initialize_fsm(restaurant_searcher, model, restaurant_manager, use_asr, use_tts, confirm_matches)
    
    while fsm.is_active:
        fsm.step()
    
    # Save the transcript at the end of the dialogue
    fsm.logger.save()
    print("\nDialogue ended. Returning to main menu...")


def start_simple_cli(models):
    """
    Launches an interactive command-line interface for classifying user input using trained models.
    """
    print("\n" + "-"*100)
    print("Welcome to the Interactive Classifier!".center(100) + "\n" + "-"*100)
    print("Enter a sentence to classify it with the selected model.")
    print("Type '!menu' to select a model, or '!quit' to exit." + "\n" + "-"*100)

    model_names = list(models.keys())
    current_model_index = None
    current_model_name = "No model selected"

    while True:
        user_input = input(f"({current_model_name}) > ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["!menu", "!model", "!switch", "!mode"]:
            print("\n--- Model Selection ---")
            for i, name in enumerate(model_names):
                print(f"  {i+1}. {name}")
            print(f"  {len(model_names)+1}. All Models (Comparison Mode)")
            print("-----------------------")
            
            try:
                choice_str = input("Enter your choice: ").strip()
                if not choice_str:
                    continue
                
                choice_index = int(choice_str) - 1

                if 0 <= choice_index < len(model_names):
                    current_model_index = choice_index
                    current_model_name = model_names[current_model_index]
                    print(f"Model set to '{current_model_name}'.")
                elif choice_index == len(model_names):
                    current_model_index = choice_index
                    current_model_name = "All Models (Comparison Mode)"
                    print(f"Model set to '{current_model_name}'.")
                else:
                    print("Invalid choice.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter a number from the list.")
            continue

        elif user_input.lower() in ["!quit", "!exit", "!escape"]:
            print("\nReturning to main menu...")
            break
        
        elif user_input.startswith("!"):
            print(f"Unknown command: '{user_input}'. Type '!menu' to select a model or '!quit' to exit.")
            continue

        if current_model_index is None:
            print("No model selected. Type '!menu' to choose one first.")
            continue

        sentence = user_input.lower()

        print("\n--------------- Prediction Results ---------------")
        if current_model_index < len(model_names):
            model = models[current_model_name]
            prediction = model.predict([sentence])[0]
            print(f"Input: '{user_input}'")
            print(f"Model: {current_model_name}")
            print(f"Predicted Act: '{prediction}'")
        else:
            print(f"Input: '{user_input}'")
            print("-" * 50)
            for name, model in models.items():
                prediction = model.predict([sentence])[0]
                print(f"{name:<25} -> '{prediction}'")
        print("-" * 50)

def start_cli(models, restaurant_manager, restaurant_searcher):
    """
    Main CLI entry point that allows switching between the simple classifier and the dialogue system.
    """
    while True:
        print("\n" + "-"*100)
        print("Main Menu".center(100))
        print("-"*100)
        print("1. Interactive Classifier (Test dialogue act models)")
        print("2. Restaurant Dialogue System")
        print("3. Exit")
        print("-"*100)
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            start_simple_cli(models)
        elif choice == '2':
            print("\n--- Select a Model for the Dialogue System ---")
            model_names = list(models.keys())
            for i, name in enumerate(model_names):
                print(f"  {i+1}. {name}")
            print("------------------------------------------")
            
            try:
                model_choice_str = input("Enter your choice: ").strip()
                if not model_choice_str:
                    continue
                
                model_choice_index = int(model_choice_str) - 1

                if 0 <= model_choice_index < len(model_names):
                    chosen_model_name = model_names[model_choice_index]
                    chosen_model = models[chosen_model_name]
                    print(f"(Using '{chosen_model_name}' for dialogue act classification)")

                    # Ask user if they want to use ASR
                    asr_choice = input("Enable ASR (Speech-to-Text)? (y/n): ").strip().lower()
                    use_asr = asr_choice == 'y'

                    # Ask user if they want to use TTS
                    tts_choice = input("Enable TTS (Text-to-Speech)? (y/n): ").strip().lower()
                    use_tts = tts_choice == 'y'

                    # Ask user if they want to confirm extracted preference matches
                    cf_choice = input("Confirm preference matches? (y/n): ").strip().lower()
                    confirm_matches = cf_choice == 'y'

                    start_dialogue_system(chosen_model, restaurant_manager, restaurant_searcher, use_asr, use_tts, confirm_matches)
                else:
                    print("Invalid choice. Returning to main menu.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter a number from the list.")

        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

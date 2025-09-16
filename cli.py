
def start_cli(models):
    """
    Launches an interactive command-line interface for classifying user input using trained models.

    Args:
        models (dict): A dictionary mapping model names (str) to trained model objects.
                       Each model must implement a .predict([text]) method.

    Features:
        - Allows the user to select a model for classification or compare predictions from all models.
        - Accepts user input sentences and displays the predicted class label.
        - Special commands:
            !menu : Open model selection menu.
            !quit : Exit the CLI.
        - Provides a user-friendly prompt and instructions.
    """

    # Intro
    print("\n" + "-"*100)
    print("Welcome to the Interactive Classifier!".center(100) + "\n" + "-"*100)
    print("Enter a sentence to classify it with the selected model.")
    print("Type '!menu' to select a model, or '!quit' to exit." + "\n" + "-"*100)

    # Get model names
    model_names = list(models.keys())
    
    # Initialization of model selection
    current_model_index = None
    current_model_name = "No model selected"

    # Keeps session active
    while True:
        user_input = input(f"({current_model_name}) > ").strip()

        if not user_input:
            continue

        # Conditionally show model selection menu
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
                
                # Use input to update the selected mode
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

        # Conditionally exit the CLI
        elif user_input.lower() in ["!quit", "!exit", "!escape"]:
            print("\nGoodbye!")
            break
        
        # Hints
        elif user_input.startswith("!"):
            print(f"Unknown command: '{user_input}'. Type '!menu' to select a model or '!quit' to exit.")
            continue

        # At this point, user_input is a sentence to classify
        if current_model_index is None:
            print("No model selected. Type '!menu' to choose one first.")
            continue

        # Input needs to be lowercased before its used for prediction
        sentence = user_input.lower()

        print("\n--------------- Prediction Results ---------------")
        if current_model_index < len(model_names):
            # Single model prediction
            model = models[current_model_name]
            prediction = model.predict([sentence])[0]
            print(f"Input: '{user_input}'")
            print(f"Model: {current_model_name}")
            print(f"Predicted Act: '{prediction}'")
        else:
            # Comparison Mode
            print(f"Input: '{user_input}'")
            print("-" * 50)
            for name, model in models.items():
                prediction = model.predict([sentence])[0]
                print(f"{name:<25} -> '{prediction}'")
        print("-" * 50)

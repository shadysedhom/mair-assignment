import random
import asyncio
import time
import wave
import threading
import pyaudio
import audioop
import os
from faster_whisper import WhisperModel
import edge_tts
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from colorama import Fore, Style, init
from playsound import playsound

from dialogue_system.finite_state_machine import FSM, State, Transition, Context, Inform, Affirm, Deny, Hello, Null,Negate
from dialogue_system import keyword_searcher
from dialogue_system.restaurant_manager import RestaurantManager

# --- ASR and TTS Helper Functions ---

# Initialize colorama
init(autoreset=True)

# ASR settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 300
SILENT_CHUNKS = 2 * (RATE // CHUNK)
AUDIO_DIR = "audio"

# Load ASR model once
asr_model = WhisperModel("base", device="cpu", compute_type="int8")

# TTS settings
VOICE = "en-US-AvaNeural"

def get_user_input(fsm: FSM) -> str:
    """Gets user input from microphone if ASR is enabled, otherwise from text prompt."""
    if not fsm.use_asr:
        return input("You: ")

    temp_wav_file = os.path.join(AUDIO_DIR, f"temp_recording_{time.time()}.wav")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print(Fore.GREEN + "[Listening...]")
    
    frames = []
    silent_chunks = 0
    is_speaking = False
    
    # Continuously record audio from the microphone until silence is detected.
    while True:
        try:
            data = stream.read(CHUNK)
            frames.append(data)

            # A simple energy-based silence detection.
            rms = audioop.rms(data, 2)
            
            if rms > SILENCE_THRESHOLD:
                is_speaking = True
                silent_chunks = 0
            elif is_speaking:
                silent_chunks += 1

            if is_speaking and silent_chunks > SILENT_CHUNKS:
                break
        except IOError:
            break

    print(Fore.BLUE + "[Processing...]")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio frames to a temporary WAV file.
    with wave.open(temp_wav_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # Transcribe the audio file using the pre-loaded Whisper model.
    segments, _ = asr_model.transcribe(temp_wav_file, beam_size=5)
    os.remove(temp_wav_file) # Clean up the temporary file
    transcribed_text = " ".join([segment.text for segment in segments]).strip()
    
    print(f"You: {transcribed_text}")
    return transcribed_text

async def _generate_and_play_tts(text: str):
    """Generates TTS audio to a unique temp file and plays it with a progress bar."""
    # Generate a unique temporary filename to avoid file lock issues.
    temp_audio_file = os.path.join(AUDIO_DIR, f"temp_tts_{time.time()}.mp3")

    # Generate the speech audio from the text using edge-tts.
    communicate = edge_tts.Communicate(text, VOICE)
    with open(temp_audio_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])

    # Estimate the audio duration (using WPM as a proxy), to make the progress bar accurate.
    words_per_minute = 150
    words = len(text.split())
    duration = (words / words_per_minute) * 60

    # Run audio playback in a separate thread to prevent blocking the progress bar.
    playback_thread = threading.Thread(target=playsound, args=(temp_audio_file,))
    playback_thread.start()

    # Show the TTS progress bar
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Speaking...", total=duration)
        start_time = time.time()
        while playback_thread.is_alive():
            elapsed = time.time() - start_time
            progress.update(task, completed=min(elapsed, duration))
            time.sleep(0.1)
        progress.update(task, completed=duration)

    playback_thread.join()

    # Clean up the temporary audio file after playback.
    os.remove(temp_audio_file)
    # Add a newline for CLI readability
    print()

def output_system_response(fsm: FSM, text: str):
    """Outputs the system's response as text and optionally as speech."""
    print(f"System: {text}")
    if fsm.use_tts:
        try:
            asyncio.run(_generate_and_play_tts(text))
        except Exception as e:
            print(Fore.RED + f"[TTS Error] Could not play audio: {e}")

# --- FSM Initialization and Actions ---

food_preference_hints = [
    "Some popular options are 'italian', 'chinese', or 'indian'.",
    "For example, you could try 'french', 'thai', or 'vietnamese'.",
    "You could also choose something like 'british', 'seafood', or 'gastropub'."
]

def initialize_fsm(keyword_searcher: keyword_searcher, ML_model, restaurant_manager: RestaurantManager, use_asr: bool, use_tts: bool, confirm_matches: bool = False) -> FSM:

    def _tokenize(text: str):
        return [t for t in ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in text.lower()).split() if t]

    def _confirm_term(fsm: FSM, attribute: str, term: str) -> bool:
        print(f"System: Did you mean '{term}' for {attribute}? (y/n)")
        resp = get_user_input(fsm).strip().lower()
        return resp in ["y", "yes"]

    def _process_preferences(fsm: FSM, text_input: str):
        """Helper to extract all preferences from a user utterance and update the context."""
        area_output = fsm.keyword_searcher.search(text_input, "area")
        food_output = fsm.keyword_searcher.search(text_input, "food")
        pricerange_output = fsm.keyword_searcher.search(text_input, "pricerange")

        if area_output:
            if not fsm.confirm_matches or _confirm_term(fsm, "area", area_output):
                fsm.context.area_known = True
                fsm.context.area = area_output
        if food_output:
            if not fsm.confirm_matches or _confirm_term(fsm, "food", food_output):
                fsm.context.food_known = True
                fsm.context.food = food_output
        if pricerange_output:
            if not fsm.confirm_matches or _confirm_term(fsm, "pricerange", pricerange_output):
                fsm.context.pricerange_known = True
                fsm.context.pricerange = pricerange_output
        
        return area_output, food_output, pricerange_output

    def welcome_action(fsm: FSM):
        output_system_response(fsm, "Welcome! Let's start. What kind of restaurant are you looking for? Please inform me about your preferences (area, food, price range).")
        text = get_user_input(fsm)
        area_found, food_found, pricerange_found = _process_preferences(fsm, text)
        return fsm.ML_model.predict([text])[0]

    def ask_area_action(fsm: FSM):
        valid_options = fsm.restaurant_manager.get_labels('area')
        output_system_response(fsm, "Which area would you like?")
        text_input = get_user_input(fsm)
        area_found, food_found, pricerange_found = _process_preferences(fsm, text_input)

        if not area_found:
            hint_options = ', '.join([opt for opt in valid_options if opt])
            output_system_response(fsm, f"I'm sorry, I don't recognize that area. Please choose from: {hint_options}.")

        return fsm.ML_model.predict([text_input])[0]

    def ask_food_action(fsm: FSM): 
        output_system_response(fsm, "What type of food do you prefer?")
        text_input = get_user_input(fsm)
        area_found, food_found, pricerange_found = _process_preferences(fsm, text_input)

        if not food_found:
            output_system_response(fsm, random.choice(food_preference_hints))

        return fsm.ML_model.predict([text_input])[0]

    def ask_pricerange_action(fsm: FSM):
        valid_options = fsm.restaurant_manager.get_labels('pricerange')
        output_system_response(fsm, "What price range are you looking for?")
        text_input = get_user_input(fsm)
        area_found, food_found, pricerange_found = _process_preferences(fsm, text_input)

        if not pricerange_found:
            hint_options = ', '.join([opt for opt in valid_options if opt])
            output_system_response(fsm, f"I'm sorry, I don't recognize that price range. Please choose from: {hint_options}.")

        return fsm.ML_model.predict([text_input])[0]

    def suggest_restaurant_action(fsm: FSM):
        matches = fsm.restaurant_manager.find_restaurants(
            area=fsm.context.area,
            pricerange=fsm.context.pricerange,
            food=fsm.context.food
        )

        if not matches:
            output_system_response(fsm, "I'm sorry, there are no restaurants that match your request.")
            return "none" 

        suggestion = random.choice(matches)
        fsm.context.remaining_matches = [r for r in matches if r != suggestion]

        output_system_response(fsm, f"{suggestion.name} is a nice place in the {suggestion.area} part of town serving {suggestion.food} food in the {suggestion.pricerange} price range.")
        return "inform"
    
    def ask_conformation_action(fsm: FSM): 
        output_system_response(fsm, "Is this suggestion okay for you?")
        text_input = get_user_input(fsm)
        action = fsm.ML_model.predict([text_input])[0]
        return action
    
    def ask_part_incorrect_action(fsm: FSM): 
        output_system_response(fsm, "Which part of the suggestion was incorrect? (Area, Food, Price Range, All)")
        text_input = get_user_input(fsm)
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
    
        output_system_response(fsm, "Please express your preferences again (area, food, price range).")
        text_input = get_user_input(fsm)
        area_found, food_found, pricerange_found = _process_preferences(fsm, text_input)
        return fsm.ML_model.predict([text_input])[0]

    def bye_action(fsm: FSM): 
        output_system_response(fsm, "Goodbye!")
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
    fsm = FSM(welcome, ctx, keyword_searcher, ML_model, restaurant_manager, use_asr=use_asr, use_tts=use_tts)
    # Feature toggle: confirm extracted preference matches (for non-direct matches)
    fsm.confirm_matches = bool(confirm_matches)

    return fsm

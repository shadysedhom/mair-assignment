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
from dialogue_system.reasoner import reason_about_restaurants
from dialogue_system.types import SearchThemes
from dialogue_system.response_templates import HUMANLIKE_TEMPLATES, SYSTEM_TEMPLATES

# --- ASR and TTS Helper Functions ---

init(autoreset=True)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 300
SILENT_CHUNKS = 2 * (RATE // CHUNK)
AUDIO_DIR = "audio"

asr_model = WhisperModel("base", device="cpu", compute_type="int8")

VOICE = "en-US-AvaNeural"

def get_user_input(fsm: FSM) -> str:
    if not fsm.use_asr:
        text_input = input("You: ")
        fsm.logger.log_turn("User", text_input, fsm.current_state.name)
        return text_input

    temp_wav_file = os.path.join(AUDIO_DIR, f"temp_recording_{time.time()}.wav")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print(Fore.GREEN + "[Listening...]")
    
    frames = []
    silent_chunks = 0
    is_speaking = False
    
    while True:
        try:
            data = stream.read(CHUNK)
            frames.append(data)
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

    with wave.open(temp_wav_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    segments, _ = asr_model.transcribe(temp_wav_file, beam_size=5)
    os.remove(temp_wav_file)
    transcribed_text = " ".join([segment.text for segment in segments]).strip()
    
    fsm.logger.log_turn("User", transcribed_text, fsm.current_state.name)
    print(f"You: {transcribed_text}")
    return transcribed_text

async def _generate_and_play_tts(text: str):
    temp_audio_file = os.path.join(AUDIO_DIR, f"temp_tts_{time.time()}.mp3")
    communicate = edge_tts.Communicate(text, VOICE)
    with open(temp_audio_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])

    words_per_minute = 150
    words = len(text.split())
    duration = (words / words_per_minute) * 60

    playback_thread = threading.Thread(target=playsound, args=(temp_audio_file,))
    playback_thread.start()

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
    os.remove(temp_audio_file)
    print()

def output_system_response(fsm: FSM, template_key: str, **kwargs):
    if fsm.response_mode == "humanlike":
        template = HUMANLIKE_TEMPLATES.get(template_key, "Error: Template not found.")
    else:
        template = SYSTEM_TEMPLATES.get(template_key, "Error: Template not found.")

    if callable(template):
        text = template()
    else:
        text = template.format(**kwargs)

    print(f"System: {text}")
    fsm.logger.log_turn("System", text, fsm.current_state.name)
    if fsm.use_tts:
        try:
            asyncio.run(_generate_and_play_tts(text))
        except Exception as e:
            print(Fore.RED + f"[TTS Error] Could not play audio: {e}")

# --- FSM Initialization and Actions ---

def initialize_fsm(keyword_searcher: keyword_searcher, ML_model, restaurant_manager: RestaurantManager, use_asr: bool, use_tts: bool, confirm_matches: bool = False, response_mode: str = "humanlike") -> FSM:

    def _tokenize(text: str):
        return [t for t in ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in text.lower()).split() if t]

    def _confirm_term(fsm: FSM, attribute: str, term: str) -> bool:
        output_system_response(fsm, "confirm_term", term=term, attribute=attribute)
        resp = get_user_input(fsm).strip().lower()
        return resp in ["y", "yes"]

    def _process_preferences(fsm: FSM, text_input: str):
        area_output = fsm.keyword_searcher.search(text_input, SearchThemes.area)
        food_output = fsm.keyword_searcher.search(text_input, SearchThemes.food)
        pricerange_output = fsm.keyword_searcher.search(text_input, SearchThemes.pricerange)

        if area_output:
            if not fsm.context.restaurants_matches or _confirm_term(fsm, "area", area_output):
                fsm.context.area_known = True
                fsm.context.area = area_output
        if food_output:
            if not fsm.context.restaurants_matches or _confirm_term(fsm, "food", food_output):
                fsm.context.food_known = True
                fsm.context.food = food_output
        if pricerange_output:
            if not fsm.context.restaurants_matches or _confirm_term(fsm, "pricerange", pricerange_output):
                fsm.context.pricerange_known = True
                fsm.context.pricerange = pricerange_output
        
        return area_output, food_output, pricerange_output
    
    def _extra_process_preferences(fsm: FSM, text_input: str):
        touristic_output = fsm.keyword_searcher.search(text_input, SearchThemes.touristic)
        assigned_seats_output = fsm.keyword_searcher.search(text_input, SearchThemes.assigned_seats)
        children_output = fsm.keyword_searcher.search(text_input, SearchThemes.children)
        romantic_output = fsm.keyword_searcher.search(text_input, SearchThemes.romantic)

        is_touristic = None
        is_assigned_seats = None
        has_children = None
        is_romantic = None

        if touristic_output:
            if not fsm.context.restaurants_matches or _confirm_term(fsm, "touristic", touristic_output):
                is_touristic = True
        if assigned_seats_output:
            if not fsm.context.restaurants_matches or _confirm_term(fsm, "assigned seats", assigned_seats_output):
                is_assigned_seats = True
        if children_output:
            if not fsm.context.restaurants_matches or _confirm_term(fsm, "children", children_output):
                has_children = True
        if romantic_output:
            if not fsm.context.restaurants_matches or _confirm_term(fsm, "romantic", romantic_output):
                is_romantic = True
        return is_touristic, is_assigned_seats, has_children, is_romantic

    def welcome_action(fsm: FSM):
        output_system_response(fsm, "welcome")
        text = get_user_input(fsm)
        _process_preferences(fsm, text)
        return fsm.ML_model.predict([text])[0]

    def ask_area_action(fsm: FSM):
        valid_options = fsm.restaurant_manager.get_labels('area')
        output_system_response(fsm, "ask_area")
        while True:
            text_input = get_user_input(fsm)
            area_found, _, _ = _process_preferences(fsm, text_input)
            if area_found:
                break
            else:
                output_system_response(fsm, "ask_area_invalid", hint_options=', '.join([opt for opt in valid_options if opt]))

        return fsm.ML_model.predict([text_input])[0]

    def ask_food_action(fsm: FSM): 
        output_system_response(fsm, "ask_food")
        text_input = get_user_input(fsm)
        _, food_found, _ = _process_preferences(fsm, text_input)

        if not food_found:
            output_system_response(fsm, "ask_food_invalid")

        return fsm.ML_model.predict([text_input])[0]

    def ask_pricerange_action(fsm: FSM):
        valid_options = fsm.restaurant_manager.get_labels('pricerange')
        output_system_response(fsm, "ask_pricerange")
        text_input = get_user_input(fsm)
        _, _, pricerange_found = _process_preferences(fsm, text_input)

        if not pricerange_found:
            output_system_response(fsm, "ask_pricerange_invalid", hint_options=', '.join([opt for opt in valid_options if opt]))

        return fsm.ML_model.predict([text_input])[0]

    def suggest_restaurant_action(fsm: FSM):
        if not fsm.context.restaurants_matches:
            output_system_response(fsm, "no_results")
            return "none" 

        suggestion = random.choice(fsm.context.restaurants_matches)
        fsm.context.restaurants_matches = [r for r in fsm.context.restaurants_matches if r != suggestion]

        output_system_response(fsm, "suggest_restaurant", name=suggestion.name, area=suggestion.area, food=suggestion.food, pricerange=suggestion.pricerange)
        return "inform"
    
    def ask_conformation_action(fsm: FSM): 
        output_system_response(fsm, "ask_conformation")
        text_input = get_user_input(fsm)
        action = fsm.ML_model.predict([text_input])[0]
        return action
    
    def ask_part_incorrect_action(fsm: FSM): 
        output_system_response(fsm, "ask_part_incorrect")
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
    
        output_system_response(fsm, "ask_preference_again")
        text_input = get_user_input(fsm)
        _process_preferences(fsm, text_input)
        return fsm.ML_model.predict([text_input])[0]

    def bye_action(fsm: FSM): 
        output_system_response(fsm, "bye")
        fsm.is_active = False
        return "bye"
    
    def show_possible_restaurants_action(fsm: FSM):
        matches = fsm.restaurant_manager.find_restaurants(
            area=fsm.context.area,
            pricerange=fsm.context.pricerange,
            food=fsm.context.food
        )

        fsm.context.restaurants_matches = matches

        if not matches:
            output_system_response(fsm, "no_results")
            return "none" 
        else:
            output_system_response(fsm, "show_possible_restaurants_count", count=len(matches))
            for r in matches:
                output_system_response(fsm, "show_restaurant_details", name=r.name, food=r.food, pricerange=r.pricerange, area=r.area)

        return "inform"     
    
    def ask_extra_preference_action(fsm: FSM):
        output_system_response(fsm, "ask_extra_preference")
        text_input = get_user_input(fsm)

        is_touristic, is_assigned_seats, has_children, is_romantic = _extra_process_preferences(fsm, text_input)

        if not any([is_touristic, is_assigned_seats, has_children, is_romantic]):
            return "affirm"

        fsm.context.restaurants_matches = reason_about_restaurants(
            fsm.context.restaurants_matches,
            touristic=is_touristic,
            assigned_seats=is_assigned_seats,
            children=has_children,
            romantic=is_romantic
        )

        return "inform"

    welcome = State("welcome", welcome_action)
    ask_area = State("ask_area", ask_area_action)
    ask_food = State("ask_food", ask_food_action)
    ask_pricerange = State("ask_pricerange", ask_pricerange_action)
    suggest_restaurant = State("suggest_restaurant", suggest_restaurant_action)
    ask_conformation = State("ask_conformation", ask_conformation_action)
    ask_part_incorrect = State("ask_part_incorrect", ask_part_incorrect_action)
    ask_preference = State("ask_to_express_preference", ask_preference_action)
    show_possible_restaurants = State("show_possible_restaurants", show_possible_restaurants_action)
    ask_extra_preference = State("ask_extra_preference", ask_extra_preference_action)   
    bye = State("bye", bye_action)

    # Reordered transitions for welcome state
    welcome.add_transition(Transition(show_possible_restaurants, lambda a, c: isinstance(a, (Inform, Hello, Null)) and c.area_known and c.food_known and c.pricerange_known))
    welcome.add_transition(Transition(ask_food, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.food_known))
    welcome.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.pricerange_known))
    welcome.add_transition(Transition(ask_area, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.area_known))

    # Reordered transitions for ask_area state
    ask_area.add_transition(Transition(show_possible_restaurants, lambda a, c: isinstance(a, Inform) and c.area_known and c.food_known and c.pricerange_known))
    ask_area.add_transition(Transition(ask_food, lambda a, c: isinstance(a, Inform) and c.area_known and not c.food_known))
    ask_area.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, Inform) and c.area_known and not c.pricerange_known))
    ask_area.add_transition(Transition(ask_area, lambda a, c: isinstance(a, Inform) and not c.area_known))

    # Reordered transitions for ask_food state
    ask_food.add_transition(Transition(show_possible_restaurants, lambda a, c: isinstance(a, Inform) and c.area_known and c.food_known and c.pricerange_known))
    ask_food.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, Inform) and c.food_known and not c.pricerange_known))
    ask_food.add_transition(Transition(ask_area, lambda a, c: isinstance(a, Inform) and c.food_known and not c.area_known))
    ask_food.add_transition(Transition(ask_food, lambda a, c: isinstance(a, Inform) and not c.food_known))

    # Reordered transitions for ask_pricerange state
    ask_pricerange.add_transition(Transition(show_possible_restaurants, lambda a, c: isinstance(a, Inform) and c.area_known and c.food_known and c.pricerange_known))
    ask_pricerange.add_transition(Transition(ask_food, lambda a, c: isinstance(a, Inform) and c.pricerange_known and not c.food_known))
    ask_pricerange.add_transition(Transition(ask_area, lambda a, c: isinstance(a, Inform) and c.pricerange_known and not c.area_known))
    ask_pricerange.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, Inform) and not c.pricerange_known))

    ask_conformation.add_transition(Transition(bye, lambda a, c: isinstance(a, Affirm)))
    ask_conformation.add_transition(Transition(ask_part_incorrect, lambda a, c: isinstance(a, (Deny, Negate))))
    ask_conformation.add_transition(Transition(ask_conformation, lambda a, c: not isinstance(a, (Affirm, Deny))))

    ask_part_incorrect.add_transition(Transition(ask_preference, lambda a, c: isinstance(a, Inform) and c.incorrect_part == "all"))
    ask_part_incorrect.add_transition(Transition(ask_area, lambda a, c: isinstance(a, Inform) and c.incorrect_part == "area"))
    ask_part_incorrect.add_transition(Transition(ask_food, lambda a, c: isinstance(a, Inform) and c.incorrect_part == "food"))
    ask_part_incorrect.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, Inform) and c.incorrect_part == "pricerange"))

    ask_preference.add_transition(Transition(show_possible_restaurants, lambda a, c: isinstance(a, (Inform, Hello, Null)) and c.area_known and c.food_known and c.pricerange_known))
    ask_preference.add_transition(Transition(ask_food, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.food_known))
    ask_preference.add_transition(Transition(ask_pricerange, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.pricerange_known))
    ask_preference.add_transition(Transition(ask_area, lambda a, c: isinstance(a, (Inform, Hello, Null)) and not c.area_known))

    def no_matches_condition(a, c):
        print(f"{len(c.restaurants_matches)} restaurants match the preferences.")
        return len(c.restaurants_matches) == 0

    ask_extra_preference.add_transition(
        Transition(ask_preference, no_matches_condition)
    )

    show_possible_restaurants.add_transition(
    Transition(ask_preference, lambda a, c: len(c.restaurants_matches) == 0)
    )
    show_possible_restaurants.add_transition(
        Transition(ask_conformation, lambda a, c: len(c.restaurants_matches) == 1)
    )
    show_possible_restaurants.add_transition(
        Transition(ask_extra_preference, lambda a, c: len(c.restaurants_matches) > 1)
    )

    ask_extra_preference.add_transition(Transition(suggest_restaurant, lambda a, c: isinstance(a, Inform)))
    ask_extra_preference.add_transition(Transition(suggest_restaurant, lambda a, c: isinstance(a, Affirm)))

    suggest_restaurant.add_transition(Transition(ask_conformation, lambda a, c: isinstance(a, Inform)))
    suggest_restaurant.add_transition(Transition(ask_preference, lambda a, c: isinstance(a, Null)))

    ctx = Context()
    fsm = FSM(welcome, ctx, keyword_searcher, ML_model, restaurant_manager, use_asr=use_asr, use_tts=use_tts, response_mode=response_mode)
    fsm.confirm_matches = bool(confirm_matches)

    return fsm

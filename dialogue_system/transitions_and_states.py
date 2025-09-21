from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import pandas as pd


class State(Enum):
    welcome = 1
    ask_area = 2
    ask_food = 3
    ask_pricerange = 4
    suggest_restaurant = 5
    ask_conformation = 6
    ask_to_express_preference = 8
    ask_which_part_incorrect = 7
    bye = 9

class Act(Enum):
    inform = auto()
    affirm = auto()
    deny = auto()
    bye = auto()
    hello = auto()
    none = auto()

@dataclass
class Context:
    state: State = State.welcome
    area_isknown: bool = False
    food_isknown : bool = False
    pricerange_isknown : bool = False

def transition(current: Context, user_act: Act) -> State:
    s = current.state

    if s == State.welcome:
        if user_act in (Act.inform, Act.hello, Act.none):
            if not current.area_isknown:
                return State.ask_area
            if not current.food_isknown:
                return State.ask_food
            if not current.pricerange_isknown:
                return State.ask_pricerange
            return State.suggest_restaurant
            
            
##ask areaa
     
    if s == State.ask_area: 
        if user_act == Act.inform:
            current.area_known = True
            if not current.food_known:
                return State.ask_food
            if not current.pricerange_known:
                return State.ask_pricerange
            return State.suggest_restaurant
        return State.ask_area 

   ##askfood
    if s == State.ask_food:
        if user_act == Act.inform:
            current.food_known = True
            if not current.pricerange_known:
                return State.ask_pricerange
            return State.suggest_restaurant
        return State.ask_food

    #askpricerange
    if s == State.ask_pricerange:
        if user_act == Act.inform:
            current.pricerange_known = True
            return State.suggest_restaurant
        return State.ask_pricerange

   #suggest restaurant
    if s == State.suggest_restaurant:
        return State.ask_conformation

 #check with user
    if s == State.ask_conformation:
        if user_act == Act.affirm:
            return State.bye
        if user_act == Act.deny:
            return State.ask_which_part_incorrect
        # unclear → re-ask confirmation
        return State.ask_conformation

   #if incorrect -> which part?
    if s == State.ask_which_part_incorrect:
        if user_act == Act.inform:
            return State.ask_to_express_preference
        return State.ask_which_part_incorrect

   #ask preferences again
    if s == State.ask_to_express_preference:
        if user_act == Act.inform:
            current.area_known = current.food_known = current.pricerange_known = False
            return State.ask_area
        return State.ask_to_express_preference
    if s == State.bye:
        return State.bye

    return s

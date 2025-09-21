from __future__ import annotations
from typing import Callable, List, Optional
from dataclasses import dataclass, field

from dialogue_system import keyword_searcher
from dialogue_system.restaurant_manager import RestaurantManager


# --- Context ---
@dataclass
class Context:
    area_known: bool = False
    food_known: bool = False
    pricerange_known: bool = False
    area : Optional[str] = None
    food : Optional[str] = None
    pricerange : Optional[str] = None
    incorrect_part : Optional[str] = None
    remaining_matches: List = field(default_factory=list)


# --- Actions ---
class Action: pass

class Acknowledge(Action): pass
class Affirm(Action): pass
class Bye(Action): pass
class Confirm(Action): pass
class Deny(Action): pass
class Hello(Action): pass
class Inform(Action): pass
class Negate(Action): pass
class Null(Action): pass
class Repeat(Action): pass
class Reqalts(Action): pass
class ReqMore(Action): pass
class Request(Action): pass
class Restart(Action): pass
class Thankyou(Action): pass

class Transition:
    def __init__(self, target: State, trigger: Callable[[Action, Context], bool]):
        self.target = target
        self.trigger = trigger

    def is_triggered(self, action: Action, context: Context) -> bool:
        return self.trigger(action, context)


from typing import Callable, List, Optional

class State:
    def __init__(self, name: str, action: Optional[Callable[["FSM"], None]] = None):
        """
        :param name: Name of the state
        :param action: A callable with signature (context, fsm) executed when the state runs
        """
        self.name = name
        self.action = action or (lambda ctx, fsm: None)
        self.transitions: List["Transition"] = []

    def add_transition(self, transition: "Transition"):
        """Add a transition from this state to another state."""
        self.transitions.append(transition)

    def run(self,fsm: "FSM"):
        """Execute the state's behavior."""
        return self.action(fsm)

    def possible_transitions(self, action: "Action", context: "Context") -> List["Transition"]:
        """Return all transitions triggered by the given action and context."""
        return [t for t in self.transitions if t.is_triggered(action, context)]

class FSM:
    def __init__(self, initial_state: State, context: Context, keyword_searcher: keyword_searcher, ML_model, restaurant_manager: RestaurantManager) -> None:
        self.current_state = initial_state
        self.context = context
        self.keyword_searcher = keyword_searcher
        self.ML_model = ML_model
        self.restaurant_manager = restaurant_manager
        self.is_active = True

    def step(self):

        string_action = self.current_state.run(self)

        action = None

        print(f"[FSM] Current state: {self.current_state.name}, Action: {string_action}")

        action = self.set_new_action(string_action)


        candidates = self.current_state.possible_transitions(action, self.context)
        if candidates:
            self.current_state = candidates[0].target
        else:
            print("[FSM] No valid transition, staying in same state.")

    def set_new_action(self, string_action):
        if string_action == "inform":
            action = Inform()
        elif string_action == "affirm":
            action = Affirm()
        elif string_action == "deny":   
            action = Deny()
        elif string_action == "hello":
            action = Hello()
        elif string_action == "bye":
            action = Bye()
        elif string_action == "none":
            action = Null()
        elif string_action == "acknowledge":
            action = Acknowledge()
        elif string_action == "confirm":
            action = Confirm()
        elif string_action == "negate":
            action = Negate()
        elif string_action == "repeat":
            action = Repeat()
        elif string_action == "reqalts":
            action = Reqalts()
        elif string_action == "reqmore":
            action = ReqMore()
        elif string_action == "request":
            action = Request()
        elif string_action == "restart":
            action = Restart()
        elif string_action == "thankyou":
            action = Thankyou()
        return action

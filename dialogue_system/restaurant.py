import random

class Restaurant:
    def __init__(self, name, pricerange, area, food, phone, addr, postcode):
        self.name = name
        self.pricerange = pricerange
        self.area = area
        self.food = food
        self.phone = phone
        self.addr = addr
        self.postcode = postcode

        self.food_quality = random.choice(["poor", "average", "good"])
        self.crowdedness = random.choice(["empty", "moderate", "busy"])
        self.length_of_stay = random.choice(["short", "medium", "long"])

    def __repr__(self):
        return f"<Restaurant {self.name} ({self.food}, {self.area}, {self.pricerange})>"
from data.csv_reader import CSVReader
from data.restaurant import Restaurant

class RestaurantReader:
    def __init__(self, filepath):
        self.csv_reader = CSVReader(filepath)

    def read_restaurants(self):
        """Reads the CSV and returns a list of Restaurant objects."""
        rows = self.csv_reader.read()
        restaurants = []
        for row in rows:
            restaurant = Restaurant(
                name=row.get("restaurantname"),
                pricerange=row.get("pricerange"),
                area=row.get("area"),
                food=row.get("food"),
                phone=row.get("phone"),
                addr=row.get("addr"),
                postcode=row.get("postcode")
            )
            restaurants.append(restaurant)
        return restaurants
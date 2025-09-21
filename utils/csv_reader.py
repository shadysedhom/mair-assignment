import csv

class CSVReader:
    def __init__(self, filepath):
        self.filepath = filepath

    def read(self):
        """Reads a CSV file and returns a list of dictionaries."""
        data = []
        with open(self.filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        return data

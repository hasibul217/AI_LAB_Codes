import random
import csv
from collections import deque

engine_file = "engines.txt"
tire_file = "tires.txt"
transmission_file = "transmissions.txt"
valid_cars_file = "valid_book.csv"

valid_cars = set()
class Car:
    def __init__(self, engine, tire, transmission, roof):
        self.transmission = transmission
        self.engine = engine
        self.tire = tire
        self.roof = roof

    def __eq__(self, other):
        return isinstance(other, Car) and self.engine == other.engine and self.tire == other.tire and self.transmission == other.transmission and self.roof == other.roof
    def __hash__(self):
        return hash((self.engine, self.tire, self.transmission, self.roof))



def load_cars(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for engine, tire, transmission, roof in reader:
            valid_cars.add(Car(engine, tire, transmission, roof))




def content_reader(filename):
    with open(filename) as file:
        container = [line.rstrip() for line in file]
        return container

engines = content_reader(engine_file)
transmission = content_reader(transmission_file)
tires = content_reader(tire_file)
roofs = ["Sunroof", "Moonroof", "Noroof"]

start_car = Car("EFI","Danlop", "AT", "Noroof")

goal_car = Car("V12", "Pirelli", "CVT", "Sunroof")

from collections import deque
frontier = deque()
level = -1
seen = set()
seen.add(start_car)
frontier.append(start_car)
goal_reached = False
while frontier:
    level += 1
    # explore the current level
    children = deque()
    while frontier:
        current_car = frontier.popleft()
        current_engine = current_car.engine
        current_tire = current_car.tire
        current_transmission = current_car.transmission
        current_roof = current_car.roof
        children = deque()
        for engine in engines:
            candidate_car = Car(engine, current_tire, current_transmission, current_roof)
            if candidate_car not in seen and candidate_car in valid_cars:
                if candidate_car == goal_car:
                    print(level + 1)
                    goal_reached = True
                    break
                children.append(candidate_car)
                seen.add(candidate_car)
        if goal_reached:
            break
    frontier = children
    if goal_reached:
        break
    if not goal_reached:
        print("Goal car could not be reached.")


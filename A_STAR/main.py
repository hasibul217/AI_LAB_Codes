import math
import time
import csv
from collections import defaultdict
from heapq import heapify, heappop, heappush

COORDINATES_FILE = "Coordinates.csv"
DISTANCE_FILE = "distances.csv"

def distance(x1, y1, z1, x2, y2, z2): 
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def dijkstra(adjacency_list, coordinates, source_star, destination_star):
    priority_queue = [(0, source_star)]
    visited = set()
    heapify(priority_queue)

    while priority_queue:
        dist, current_star = heappop(priority_queue)
        if current_star in visited:
            continue
        if current_star == destination_star:
            return dist
        visited.add(current_star)
        for neighbor_star, neighbor_dist in adjacency_list[current_star]:
            heappush(priority_queue, (dist + neighbor_dist, neighbor_star))

    return float('inf')

def a_star(adjacency_list, coordinates, source_star, destination_star):
    def heuristic(star1, star2):
        x1, y1, z1 = coordinates[star1]
        x2, y2, z2 = coordinates[star2]
        return distance(x1, y1, z1, x2, y2, z2)

    priority_queue = [(0, source_star)]
    visited = set()
    heapify(priority_queue)

    while priority_queue:
        dist, current_star = heappop(priority_queue)
        if current_star in visited:
            continue
        if current_star == destination_star:
            return dist
        visited.add(current_star)
        for neighbor_star, neighbor_dist in adjacency_list[current_star]:
            heuristic_dist = dist + neighbor_dist + heuristic(neighbor_star, destination_star)
            heappush(priority_queue, (heuristic_dist, neighbor_star))

    return float('inf')


coordinates = {}
with open(COORDINATES_FILE, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for star_name, x, y, z in reader:
        coordinates[star_name] = (int(x), int(y), int(z))


adjacency_list = defaultdict(list)
with open(DISTANCE_FILE, "r") as file:
    reader = csv.reader(file)
    for source, destination, dist in reader:
        adjacency_list[source].append((destination, int(dist)))


SOURCE_STAR = "Sun"
DESTINATION_STAR = "Upsilon Andromedae"
# DESTINATION_STAR = "61 Virginis"

# Dijkstra's
dijkstra_distance = dijkstra(adjacency_list, coordinates, SOURCE_STAR, DESTINATION_STAR)
print("Dijkstra's Algorithm:")
if dijkstra_distance != float('inf'):
    print("Shortest path from", SOURCE_STAR, "to", DESTINATION_STAR, ":", dijkstra_distance)
else:
    print("No path found from", SOURCE_STAR, "to", DESTINATION_STAR)

# A*
a_star_distance = a_star(adjacency_list, coordinates, SOURCE_STAR, DESTINATION_STAR)
print("A* Algorithm:")
if a_star_distance != float('inf'):
    print("Shortest path from", SOURCE_STAR, "to", DESTINATION_STAR, ":", a_star)
else:
    print("No path found from", SOURCE_STAR, "to", DESTINATION_STAR)


start = time.time()

a = 0
for i in range(1000):
    a += (i ** 100)
end = time.time()
print("The time of execution of above program is :",
      (end - start) * 10 ** 3, "ms")


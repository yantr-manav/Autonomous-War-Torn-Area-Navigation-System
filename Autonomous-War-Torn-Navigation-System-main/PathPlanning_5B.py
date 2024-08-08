import heapq
from collections import deque
import serial
import csv
import time
import numpy as np
import cv2

# VARIABLE NAMES

send = []
ar_id = []
lat = []
long = []
coordinate = []

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# VARIABLE NAMES

dictionary = {
    'A': 'humanitarian_aid_rehab',
    'B': 'destroyed_buildings',
    'C': 'combat',
    'D': 'military_vehicles',
    'E': 'fire'
}

nodes = {'a': 6, 'b': 6, 'c': 6, 'd': 6, 'e': 6, 'f': 6,
         'g': 6, 'h': 6, 'i': 6, 'j': 6, 'k': 6, 'l': 0}

send = []

adjacency_matrix = [
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

######################################################

def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
       
def calculate_distance(marker1, marker2):
    centroid1 = np.mean(marker1, axis=1).flatten()
    centroid2 = np.mean(marker2, axis=1).flatten()
    distance = np.linalg.norm(centroid1 - centroid2)
    return distance


def read_csv(csv_name):
    lat_lon = {}

    # open csv file (lat_lon.csv)
    # read "lat_lon.csv" file
    # store csv data in lat_lon dictionary as {id: [header1, header2, ...], ...}
    # return lat_lon

    with open(csv_name, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read and store the header row

        lat_lon[header[0]] = header[1:]
        for row in reader:
            id = str(row[0])
            lat = str(row[1])
            lon = str(row[2])
            lat_lon[id] = [lat, lon]

    return lat_lon


def write_csv(loc, csv_name):
    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['lat', 'lon'])
        writer.writerow(loc)


def write_full_csv(loc, csv_name):
    with open(csv_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(loc)


def tracker(ar_id, lat_lon, id):
    for id, lat_lon_val in lat_lon.items():
        lat.append(lat_lon_val[0]) 
        long.append(lat_lon_val[1])  

    lat_lon_val_new = lat_lon[f"{ar_id[0]}"]
    path = 'C:/Yantrika/GG_Task5_1709/live_location.csv'
    write_csv(lat_lon_val_new, path)

    coordinate_val = lat_lon.get(id)
    coordinate.append(coordinate_val)

    return coordinate


def main():
    lat_lon = read_csv('C:/Yantrika/GG_Task5_1709/lat_long.csv')
    traversedPath1 = []
    id_lel = 0

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_4X4_250"])
    arucoParams = cv2.aruco.DetectorParameters()
    reference_marker_id = 100 
    reference_marker = None
    nearest_marker = None

    while True:

        ret, frame = video.read()

        if ret is True:

            h, w, _ = frame.shape

            width = 1000
            height = int(width * (h / w))
            frame = cv2.resize(frame, (width, height),
                               interpolation=cv2.INTER_CUBIC)
            corners, ids, rejected = cv2.aruco.detectMarkers(
                frame, arucoDict, parameters=arucoParams)

            detected_markers = aruco_display(corners, ids, rejected, frame)
            detected_markers = cv2.resize(
                frame, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Video", detected_markers)

            for i in range(len(ids)):
                print(ids)
                if ids[i] == reference_marker_id:
                    reference_marker = corners[i]

            if reference_marker is not None:
                nearest_distance = float('inf')

                for i in range(len(ids)):
                    if ids[i] != reference_marker_id:
                        distance = calculate_distance(
                            reference_marker, corners[i])
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest_marker = ids[i]
                print('')
                print(nearest_marker)
                print('')

            id_lel = nearest_marker
            t_point = tracker(id_lel, lat_lon, id_lel)
            traversedPath1.append(t_point)
            time.sleep(0.5)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    video.release()


######################################################
    
def priority_val(str_val):
    num = 0
    if str_val == 'fire':
        num = 1

    if str_val == 'destroyed_buildings':
        num = 2

    if str_val == 'humanitarian_aid_rehab':
        num = 3

    if str_val == 'military_vehicles':
        num = 4

    if str_val == 'combat':
        num = 5

    return num


def priority_key(key):
    if key == 'A':
        value = priority_val(dictionary['A'])
        nodes.update({'a': value, 'b': value})

    elif key == 'B':
        value = priority_val(dictionary['B'])
        nodes.update({'e': value, 'd': value})
        nodes.update({'c': value+1.5})

    elif key == 'C':
        value = priority_val(dictionary['C'])
        nodes.update({'g': value, 'h': value})

    elif key == 'D':
        value = priority_val(dictionary['D'])
        nodes.update({'f': value})

    elif key == 'E':
        value = priority_val(dictionary['E'])
        nodes.update({'i': value, 'k': value})


def create_edge_dict(adj_matrix, node_values):
    edge_dict = {}

    for i in range(len(adj_matrix)):
        for j in range(i + 1, len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:
                node1 = chr(97 + i)
                node2 = chr(97 + j)
                edge_dict.setdefault(node1, {})[node2] = abs(
                    node_values[node1] + node_values[node2]) / 2
                edge_dict.setdefault(node2, {})[node1] = abs(
                    node_values[node1] + node_values[node2]) / 2

    return edge_dict


def dijkstra(graph, start, target):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0

    paths = {node: [] for node in graph}
    paths[start] = [start]

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        # Iterate through neighbors and update distances and paths
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(priority_queue, (distance, neighbor))

    # If target is not reachable, return None
    if distances[target] == float('infinity'):
        return None, None

    return distances[target], paths[target]


def final_edge(path_list, visited_nodes):
    path = 0

    edge_check = [visited_nodes[-2], visited_nodes[-1]]

    if edge_check == ['f', 'g'] or edge_check == ['g', 'f'] or edge_check == ['g', 'h'] or edge_check == ['h', 'g']:
        print(f'Edge Check: {edge_check}')
        return visited_nodes[-1], edge_check, None

    source_node = path_list[-1]
    print(f'Source Node: {source_node}')

    inner_dict = graph.get(source_node)
    if visited_nodes[-2] in inner_dict:
        inner_dict.pop(visited_nodes[-2])
    print(f'Visited Nodes: {visited_nodes}')

    next_nodes = inner_dict.values()
    sorted_nodes = sorted(next_nodes)
    print(f"Next nodes: {sorted_nodes}, inner: {inner_dict}")

    value_of_source_node = nodes.get(source_node)
    print(f'Value of source node: {value_of_source_node}')

    only_neighbours = []

    for value in sorted_nodes:
        if value == value_of_source_node:
            only_neighbours.append(value)

            min_value = min(only_neighbours)

            target_node = next(
                (key for key, value in inner_dict.items() if value == min_value), None)

            last_edge = [source_node, target_node]

            path = target_node
            print(f"Path in Sai: {path}")

            break

    # Iterate through the values of neighboring nodes
    if path == 0:
        for value in sorted_nodes:
            print(f"Value: {value}")
            if value <= value_of_source_node:
                # Append the values that meet the condition to the list
                only_neighbours.append(value)

                # Find the minimum value among the neighbors
                min_value = min(only_neighbours)

                # Find the key(s) corresponding to the minimum value in the nodes dictionary
                target_node = next(
                    (key for key, value in inner_dict.items() if value == min_value), None)

                # Form the path from the source node to the target node
                last_edge = [source_node, target_node]

                # Print the path
                path = target_node
                print(f"Path in Sai: {path}")

                break

    return path, last_edge, 1


def traverse_nodes(node_list, certain_event_edge, num_events):
    counter = 0
    for i in range(len(node_list) - 1):
        previous_node = node_list[i - 1]
        current_node = node_list[i]
        next_node = node_list[i + 1]
        edge_check = [current_node, next_node]

        if current_node == 'L':
            if next_node == 'A':
                send.append('f')

        if current_node == 'A':
            if previous_node == 'L' and next_node == 'B':
                send.append('r')
            elif previous_node == 'L' and next_node == 'C':
                send.append('f')
            elif previous_node == 'B' and next_node == 'C':
                send.append('r')
            elif previous_node == 'C' and next_node == 'B':
                send.append('l')
            if previous_node == 'B' and next_node == 'L':
                send.append('l')
            elif previous_node == 'C' and next_node == 'L':
                send.append('f')

        elif current_node == 'B':
            if previous_node == 'A' and next_node == 'D':
                send.append('l')
            elif previous_node == 'D' and next_node == 'A':
                send.append('r')
            elif previous_node == 'A' and next_node == 'A':
                send.append('b')

        elif current_node == 'C':
            if previous_node == 'A' and next_node == 'F':
                send.append('f')
            elif previous_node == 'A' and next_node == 'D':
                send.append('r')
            elif previous_node == 'F' and next_node == 'A':
                send.append('f')
            elif previous_node == 'D' and next_node == 'A':
                send.append('l')

        elif current_node == 'D':
            if previous_node == 'C' and next_node == 'E':
                send.append('f')
            elif previous_node == 'E' and next_node == 'C':
                send.append('f')
            elif previous_node == 'B' and next_node == 'C':
                send.append('l')
            elif previous_node == 'C' and next_node == 'B':
                send.append('r')
            elif previous_node == 'B' and next_node == 'G':
                send.append('f')
            elif previous_node == 'G' and next_node == 'B':
                send.append('f')
            elif previous_node == 'C' and next_node == 'G':
                send.append('l')
            elif previous_node == 'G' and next_node == 'C':
                send.append('r')
            elif previous_node == 'E' and next_node == 'G':
                send.append('r')
            elif previous_node == 'G' and next_node == 'E':
                send.append('l')
            elif previous_node == 'E' and next_node == 'B':
                send.append('l')
            elif previous_node == 'E' and next_node == 'E':
                send.append('b')

        elif current_node == 'E':
            if previous_node == 'D' and next_node == 'H':
                send.append('l')
            elif previous_node == 'H' and next_node == 'D':
                send.append('r')
            elif previous_node == 'D' and next_node == 'D':
                send.append('b')
            elif previous_node == 'H' and next_node == 'B':
                send.append('f')
            elif previous_node == 'B' and next_node == 'H':
                send.append('f')

        elif current_node == 'F':
            if previous_node == 'C' and next_node == 'I':
                send.append('f')
            elif previous_node == 'I' and next_node == 'C':
                send.append('f')
            elif previous_node == 'C' and next_node == 'G':
                send.append('r')
            elif previous_node == 'G' and next_node == 'C':
                send.append('l')
            elif previous_node == 'I' and next_node == 'G':
                send.append('l')
            elif previous_node == 'G' and next_node == 'I':
                send.append('r')

        elif current_node == 'G':
            if previous_node == 'D' and next_node == 'J':
                send.append('f')
            elif previous_node == 'J' and next_node == 'D':
                send.append('f')
            elif previous_node == 'D' and next_node == 'F':
                send.append('l')
            elif previous_node == 'F' and next_node == 'D':
                send.append('r')
            elif previous_node == 'D' and next_node == 'H':
                send.append('r')
            elif previous_node == 'H' and next_node == 'D':
                send.append('l')
            elif previous_node == 'F' and next_node == 'H':
                send.append('f')
            elif previous_node == 'F' and next_node == 'F':
                send.append('b')
            elif previous_node == 'H' and next_node == 'H':
                send.append('b')

        elif current_node == 'H':
            if previous_node == 'E' and next_node == 'K':
                send.append('f')
            elif previous_node == 'K' and next_node == 'E':
                send.append('f')
            elif previous_node == 'E' and next_node == 'G':
                send.append('l')
            elif previous_node == 'G' and next_node == 'E':
                send.append('r')
            elif previous_node == 'K' and next_node == 'G':
                send.append('r')
            elif previous_node == 'G' and next_node == 'K':
                send.append('l')
            elif previous_node == 'G' and next_node == 'G':
                send.append('b')

        elif current_node == 'I':
            if previous_node == 'F' and next_node == 'K':
                send.append('f')
            elif previous_node == 'K' and next_node == 'F':
                send.append('f')

        elif current_node == 'K':
            if previous_node == 'I' and next_node == 'H':
                send.append('f')
            elif previous_node == 'H' and next_node == 'I':
                send.append('f')

        if counter < num_events and certain_event_edge[counter] == edge_check:
            if edge_check == ['A', 'B'] or edge_check == ['E', 'D'] or edge_check == ['F', 'G'] or edge_check == ['H', 'G']:
                    send.append('e')

            elif edge_check == ['B', 'A'] or edge_check == ['D', 'E'] or  edge_check == ['G', 'F']:
                    send.append('a')

            elif  edge_check == ['I', 'K']:
                    send.append('x')

            elif  edge_check == ['K', 'I']:
                    send.append('y')

            elif edge_check == ['G', 'H']:
                send.append('c')

            counter += 1
            print(counter)
    if  edge_check == ['A', 'L']:
        send.append('z')
    send.append('q')
    print(send)

if __name__ == "__main__":
    current_event = 0
    eventnum = 0
    event_edges_list = []
    main_path = ['l']
    last_node = 'l'

    # Giving values to nodes
    for key, val in dictionary.items():
        priority_key(key)

    if nodes['f'] < nodes['h']:
        nodes.update({'g': nodes['f']})
    print(f"Nodes: {nodes}")

    # Creating the edges and graph
    graph = create_edge_dict(adjacency_matrix, nodes)
    # Print graph dictionary
    print("graph =", graph)

    # Finding the number of events
    for key in dictionary.keys():
        eventnum += 1
        print(f"Number of Events: {eventnum}")

    # MAIN WHILE LOOP
    while (current_event <= eventnum):
        start_node = last_node
        bigger_nodes = {}
        target_node2 = 0
        current_event_value = 0

        if current_event > 0:
            if nodes[main_path[-1]] > nodes[main_path[-2]]:
                current_event_value = nodes[main_path[-1]]
            else:
                current_event_value = nodes[main_path[-2]]

        # Finding the Target Node
        for key, value in nodes.items():
            # isinstance to remove C
            if value > current_event_value and isinstance(value, int) and current_event < eventnum:
                bigger_nodes[key] = value
            if current_event == eventnum:
                bigger_nodes = nodes
        sorted_dict = dict(
            sorted(bigger_nodes.items(), key=lambda item: item[1]))
        print(f'Sorted Dict: {sorted_dict}')

        # Extracting the first and second keys using list indexing
        first_key = list(sorted_dict.values())[0]
        if len(sorted_dict) > 1:
            second_key = list(sorted_dict.values())[1]
        print(f'First Key: {first_key}, Second Key: {second_key}')

        # If both the first values are same
        if first_key == second_key:
            target_node1, target_node2 = list(sorted_dict.keys())[:2]
            print(f"Target Node 1: {target_node1}")
            print(f"Target Node 2: {target_node2}")
        else:
            # the 'iter' function is used to create an iterator from the dictionary, and the 'next' function retrieves the first key from the iterator.
            target_node1 = next(iter(sorted_dict))
            print(f"Target Node 1: {target_node1}")

        # THE SUB WHILE LOOP (For each event)
        while True:
            if target_node2 != 0:
                distance1, visited_nodes1 = dijkstra(
                    graph, start_node, target_node1)
                distance2, visited_nodes2 = dijkstra(
                    graph, start_node, target_node2)

                # Whichever one is the shortest path
                if distance1 < distance2:
                    distance, visited_nodes, target_node = distance1, visited_nodes1, target_node1
                else:
                    distance, visited_nodes, target_node = distance2, visited_nodes2, target_node2
            else:
                distance, visited_nodes = dijkstra(
                    graph, start_node, target_node1)
                target_node = target_node1

            main_path.extend(visited_nodes[1:])

            if distance is not None:
                print(
                    f"Shortest distance from {start_node} to {target_node}: {distance}")
                print(f"Path: {visited_nodes}")
            else:
                print(f"There is no path from {start_node} to {target_node}.")

            if current_event < eventnum:
                last_node, last_edge, event_occur_check = final_edge(
                    visited_nodes, visited_nodes)

                if event_occur_check != None:
                    print(f"Last Node: {last_node}")
                    main_path.append(last_node)

                event_edges_list.append(last_edge)

            print(f'Main Path: {main_path}')
            current_event += 1
            print('\n')
            break

    main_path_uppercase_list = [element.upper() for element in main_path]
    print(f'Event Edges List: {event_edges_list}')
    event_edges_list_upper = [
        [element.upper() for element in inner_list] for inner_list in event_edges_list]
    command_list = traverse_nodes(
        main_path_uppercase_list, event_edges_list_upper, eventnum)
    print(command_list)
    main()
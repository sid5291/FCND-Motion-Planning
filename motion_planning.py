import argparse
import time
import msgpack
from enum import Enum, auto
from bresenham import bresenham
from shapely.geometry import Polygon, Point, LineString
from sklearn.neighbors import KDTree
import numpy.linalg as LA
import networkx as nx
import numpy as np
from queue import PriorityQueue


from planning_utils import a_star, heuristic, create_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

class GraphPlanner(object):

    def __init__(self, data):
        self.data = data
        self.start = None
        self.goal = None
        self.polygons = dict()
        self.nodes = list()
        self.tree = list()  # KDTree for generated polygons
        self.graph = nx.Graph()
        self.extract_polygons()

    def extract_polygons(self):
        for i in range(self.data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = self.data[i, :]
            LL = ((north - d_north), (east - d_east))
            UL = ((north + d_north), (east - d_east))
            LR = ((north - d_north), (east + d_east))
            UR = ((north + d_north), (east + d_east))
            corners = [LL, LR, UR, UL]
            height = alt + d_alt
            p = Polygon(corners)
            self.polygons[(north, east)] = (p, height)
        self.tree = KDTree(list(self.polygons.keys()))

    def generate_nodes(self, max_num_nodes=1000, max_alt=10):
        xmin = np.min(self.data[:, 0] - self.data[:, 3])
        xmax = np.max(self.data[:, 0] + self.data[:, 3])

        ymin = np.min(self.data[:, 1] - self.data[:, 4])
        ymax = np.max(self.data[:, 1] + self.data[:, 4])

        zmin = 0
        zmax = max_alt

        xvals = np.random.uniform(xmin, xmax, max_num_nodes)
        yvals = np.random.uniform(ymin, ymax, max_num_nodes)
        zvals = np.random.uniform(zmin, zmax, max_num_nodes)

        self.nodes = list(zip(xvals, yvals, zvals))
        # prune Nodes
        to_keep = []
        for point in self.nodes:
            if not self.collides(point):
                to_keep.append(point)
        self.nodes = to_keep

    def collides(self, point):
        # TODO: Determine whether the point collides
        # with any obstacles.

        closest_centroids = self.tree.query([(point[0], point[1])], k=5, return_distance=False)[0]
        for centroid in closest_centroids:
            key = list(self.polygons.keys())[centroid]
            poly = self.polygons[key]
            if poly[0].contains(Point(point[0], point[1])):
                if poly[1] > point[2]:
                    return True
        return False

    def can_connect(self, point1, point2):
        line = LineString(((point1), (point2)))
        closest_centroids = self.tree.query([(point1[0], point1[1])], k=10, return_distance=False)[0]
        for centroid in closest_centroids:
            key = list(self.polygons.keys())[centroid]
            poly = self.polygons[key]
            if poly[0].crosses(line):
                return False
        closest_centroids = self.tree.query([(point2[0], point2[1])], k=10, return_distance=False)[0]
        for centroid in closest_centroids:
            key = list(self.polygons.keys())[centroid]
            poly = self.polygons[key]
            if poly[0].crosses(line):
                return False
        return True

    def find_nearest_node(self, point):
        temp = np.array(self.graph)
        near_point = temp[np.argmin(np.linalg.norm(np.array(point) - np.array(temp), axis=1))]
        return tuple(near_point)

    # In NED frame (x,y,z) or (n,e,d)
    def add_start_goal(self, start, goal):
        if not self.collides(start):
            self.nodes.append(start)
            self.start = start
        else:
            print("Start Collides")
            self.start = self.find_nearest_node(start)
            print("New Start-> {0}".format(self.start))

        if not self.collides(goal):
            self.nodes.append(goal)
            self.goal = goal
        else:
            print("Goal Collides")
            self.goal = self.find_nearest_node(goal)
            print("New Goal-> {0}".format(self.goal))

        array = np.array(self.nodes)
        points = array[..., (0, 1)]
        tree = KDTree(points)
        for point in points:
            nn = tree.query([point], k=10, return_distance=False)[0]
            for node in nn:
                if self.can_connect(point, points[node]):
                    dist = LA.norm(np.array(point) - np.array(points[node]))
                    if dist:
                        self.graph.add_edge(tuple(point), tuple(points[node]), weight=dist)
        return self.graph

    def create_graph(self):
        if len(self.nodes) == 0:
            self.generate_nodes()
        array = np.array(self.nodes)
        points = array[..., (0, 1)]
        tree = KDTree(points)
        for point in points:
            nn = tree.query([point], k=10, return_distance=False)[0]
            for node in nn:
                if self.can_connect(point, points[node]):
                    dist = LA.norm(np.array(point) - np.array(points[node]))
                    if dist:
                        self.graph.add_edge(tuple(point), tuple(points[node]), weight=dist)
        return self.graph

    def a_star(self, h):

        if (self.start is None) or (self.goal is None):
            print("ERROR: Please add the start and goal to graph using add_start_and_goal")
            return
        path = []
        path_cost = 0
        queue = PriorityQueue()
        queue.put((0, self.start))
        visited = set(self.start)

        branch = {}
        found = False

        while not queue.empty():
            item = queue.get()
            current_node = item[1]
            if current_node == self.start:
                current_cost = 0.0
            else:
                current_cost = branch[current_node][0]

            if current_node == self.goal:
                print('Found a path.')
                found = True
                break
            else:
                for neighbor in self.graph[current_node]:
                    next_node = neighbor
                    branch_cost = current_cost + self.graph[current_node][next_node]['weight']
                    queue_cost = branch_cost + h(next_node, goal)

                    if next_node not in visited:
                        visited.add(next_node)
                        branch[next_node] = (branch_cost, current_node)
                        queue.put((queue_cost, next_node))

        if found:
            n = self.goal
            path_cost = branch[n][0]
            path.append(self.goal)
            while branch[n][1] != self.start:
                path.append(branch[n][1])
                n = branch[n][1]
            path.append(branch[n][1])
        else:
            print('**********************')
            print('Failed to find a path!')
            print('**********************')
        return path[::-1], path_cost

class MotionPlanning(Drone):

    def __init__(self, connection=None):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def prune_path(self, path):
        ret = [point for point in path]
        i=0
        while i < (len(ret) - 2):
            p1 = ret[i]
            p2 = ret[i + 1]
            p3 = ret[i + 2]

            cells = bresenham(p1[0], p1[1], p3[0], p3[1])
            if p2 in cells:
                del ret[i+1]

        return ret

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5
        DEFAULT_ALTITUDE = 0.0

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        # Gives us the maps center coordinates !
        global_home_position = np.append(np.genfromtxt('colliders.csv', delimiter=' ',
                                converters={1: lambda x: float(x.decode("utf-8").replace(',', ''))}, usecols=(1, 3),
                                max_rows=1), DEFAULT_ALTITUDE)
        # TODO: set home position to (lon0, lat0, 0)
        # Ensure to invert Lat Lon as it is fed to set_home_position as Lon Lat
        self.set_home_position(global_home_position[1], global_home_position[0], global_home_position[2])
        self._longitude = global_home_position[1]
        self._latitude =  global_home_position[0]
        self._altitude =  global_home_position[2]
        # TODO: retrieve current global position
        current_global_position = self.global_position
        # TODO: convert to current local position using global_to_local()
        current_local_position = global_to_local(current_global_position, self.global_home)
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        print('Calculated local Position: {0}'.format(current_local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        graph = GraphPlanner(data)
        graph.generate_nodes(max_alt=TARGET_ALTITUDE)
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        # grid_start = (-north_offset, -east_offset)
        # TODO: convert start position to current position rather than map center
        grid_start = (int(np.floor(self.local_position[0])) - north_offset,
                      int(np.floor(self.local_position[1])) - east_offset)
        # Set goal as some arbitrary position on the grid
        grid_goal = (int(np.floor(self.local_position[0] + 3)) - north_offset,
                     int(np.floor(self.local_position[1] + 0)) - east_offset)

        graph_start = (self.local_position[0], self.local_position[1], TARGET_ALTITUDE)
        graph_goal = (self.local_position[0] + 40, self.local_position[1] + 50, TARGET_ALTITUDE)
        graph.add_start_goal(graph_start, graph_goal)
        print('Local Graph Start and Goal: ', graph_start, graph_goal)
        g = graph.create_graph()
        # TODO: adapt to set goal as latitude / longitude position and convert

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        graph_path, _ = graph.a_star(heuristic)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        # TODO: prune path to minimize number of waypoints
        path = self.prune_path(path)
        # TODO (if you're feeling ambitious): Try a different approach altogether!
        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()

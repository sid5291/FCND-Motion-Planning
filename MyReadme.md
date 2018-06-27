## Supporting Material for 3D Motion Planning Project

### Explain Starter Code:
####MotionPlanning:
1. Inherits from the Drone class, that takes a connection object to initialize the connection with the Udacidrone simulator
2. Registers the callbacks for Local_position, local_velocity and state update
3. Once the `start()` function is called the logging is enabled and connection established
4. Then the `state_callback()` will transition the drone from the MANUAL  state to the ARMED state upon which it will call the path_plan() function
####path_plan():
1. `path_plan()` currently reads in `colliders.csv` and generates a grid with all obstacles
2. It then assumes the starting position of the drone to be the center of the grid and the goal is set 10m NE of that 
3. A* is run to find the best possible path from the Start to the Goal given the current valid actions and cost
4. The waypoints are generated from the path by adding back the offsets and they are then fed to the simulator



### Implementation:

### Main Objectives:
#### Read Lat0 and Lon0
1. Starting at line 346 `colliders.csv` is parsed and the global home position `lat0` and `lon0` are read as 2 floating point values
2. The read takes care of removing the ',' and then converting `lat0` to a float
3. the global_home_position read in is then set as the home position using the set_home_position function, with a DEFAULT_ALTITUDE=0.0

#### Convert Global to Local Positon
1. The `@property global_position` provides the current global position of the drone and returns that altitude, latitude and longitude
2. The `global_to_local` helper function is used to convert the current global position to the NED frame with respect to global_home (which was set in the previous step)

### Initialize the grid/graph
1. Depending on the method selected (True-graph/False-grid) the graph/grid is initialized 
2. The graph object will extract the polygons from the data function and generate random nodes (max 1000) to be used
3. The grid is a large 2D matrix with 0's and 1's indicating feasible and unfeasible cells respectively

### Start and Goal Positions
#### Use current location as the Starting point
1. Use the current local position (from the local_position @property) to set the start position

#### Set Goal position
1. Use either a given north and east offset or a global position (in lat,lon) to set the goal position in the local frame

#### Graph Method:
1. Start and Goal Positions are also added to the list of nodes use to generate the graphs (if either collide the nearest random node is selected)

### Update A* Implementation
1. Add Valid actions for NE,SE,NW and SW enabling diagonal movement
2. Simplify logic to identify which actions are possible

### Add a Path Pruning Method
1. Created a simple `path_prune` method that simply iterates over the nodes/waypoints on the path and removes all that are collinear using Bresenhams leaving only "turning points"

### Additional Objective:
#### Probabilistic Roadmap Method:
Defined a `GraphPlanner` class that has all the required helper classes and states to produce a path from start to goal using the probabilistic method

####GraphPlanner:
1. `extract_polygons()`
    * A simple function that will take the `colliders.csv` file and extract the polygons for each center in the NED coordinates
    * It stores the polygons as `shapely.Polygon` objects in a dictionary keyed by their center point
    * Then it generates a `KDTree` for all the center points allowing us to easily search each polygon
2. `generate_nodes()`
    * This extracts the furthest corners defined in `colliders.csv` to use as the corners of the mapped area
    * Then generates uniformly distributed nodes in the NED coordinate frame totalling to `max_num_nodes` 
    * Check collision of the nodes and keep only the nodes that do not collide
3. `collides()`
    * Takes a point as a `tuple` of the form `(north, east)`
    * Use the tree to find all centers stored in the `KDTree` which are within a radius of 10m 
    * Using the dictionary of polygons the provided point is checked if it is within any of the polygons, using the `contains` method
    * `True` is returned if there is a collision
4. `can_connect`    
    * Provided with two points as tuples of the form `(north,east)` a `shapely.LineString` object is generated
    * Bresenhams is used to extract the cells that would join the two points
    * Every 10th cell(as an optimization) in the list of cells that would form the line connecting the two points is iterated over
    * The cell's coordinates are then used to query the `KDTree` of polygons to see if the line string crosses any of these polygons using the `Polygon.crosses` method
5. `find_nearest_node`
    * Simple helper function that will return the nearest node in the graph to the provided point
    * Uses the Euclidean distance
6. `add_start_goal`
    * Add the Start and Goal points to the nodes or the nearest random node if the start/goal point collide
7. `create_graph`
    * Iterates over the possible nodes and checks if the 10 nearest neighbours (extracted using the KDTree of nodes) `can_connect`
    * If the nodes can connect and edge is then added to the `networks.Graph` object, 
    * The weight for the edge is calculated as the euclidean distance between the two nodes
8. `a_start`
    * Will do a BFS of the graph starting from the start node or the closest node to the goal or the closest node
    * Uses the provided heuristic and branch cost to calculate the path cost. 
    * Returns a list of nodes on the graph representing a path from the start to the goal
    * If no path is found it will an empty path and print an error message
    
       
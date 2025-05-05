__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

# Third-party imports
import math
import numpy as np

from shapely.geometry import Point, Polygon


def ignore_vehicles_in_cone_angle(time_step, obstacles, scenario, ego_pose, veh_length, cone_angle, cone_safety_dist):
    """Ignore vehicles behind ego for prediction if inside specific cone.

    Cone is spaned from center of rear-axle (cog - length / 2.0)

    cone_angle = Totel Angle of Cone. 0.5 per side (right, left)

    return bool: True if vehicle is ignored, i.e. inside cone
    """
    ego_position = np.array([ego_pose.initial_state.position[0], ego_pose.initial_state.position[1]])
    ego_orientation = ego_pose.initial_state.orientation
    cone_angle_rad = cone_angle * np.pi / 180.0
    ignore_pred_list = []

    for i in obstacles:
        obstacle = scenario.obstacle_by_id(i)
        obj_position = obstacle.occupancy_at_time(time_step).shape.center[:2]
        # Convert object position from global to ego vehicle's local coordinate system
        loc_obj_pos = rotate_glob_loc(obj_position - ego_position, -ego_orientation, matrix=False)
        loc_obj_pos[0] -= veh_length / 2.0  # Adjust for vehicle length to consider rear axle as origin

        # Calculate the distance from the ego vehicle
        distance = np.linalg.norm(loc_obj_pos)

        # Check if the object is behind the ego vehicle, within the cone angle,
        # and farther away than the safety distance
        if loc_obj_pos[0] < 0 and distance > cone_safety_dist:
            # Calculate the angle of the object relative to the ego vehicle's orientation
            obj_angle = pi_range(math.atan2(loc_obj_pos[1], loc_obj_pos[0]))

            # Check if the object is within the cone angle
            if abs(abs(obj_angle) - np.pi) < cone_angle_rad / 2.0:
                ignore_pred_list.append(i)

    # Remove ignored obstacles from the list
    for obj in ignore_pred_list:
        obstacles.remove(obj)

    return obstacles


def get_obstacles_in_radius(scenario, ego_id: int, ego_state, time_step: int, radius: float,
                            vehicles_in_cone_angle=True, config=None):
    """
    Get all the obstacles that can be found in a given radius.

    Args:
        scenario (Scenario): Considered Scenario.
        ego_id (int): ID of the ego vehicle.
        ego_state (State): State of the ego vehicle.
        time_step (int) time step
        radius (float): Considered radius.

    Returns:
        [int]: List with the IDs of the obstacles that can be found in the ball
        with the given radius centering at the ego vehicles position.
    """
    obstacles_within_radius = []
    for obstacle in scenario.obstacles:
        # do not consider the ego vehicle
        if obstacle.obstacle_id != ego_id:
            occ = obstacle.occupancy_at_time(time_step)
            # if the obstacle is not in the lanelet network at the given time, its occupancy is None
            if occ is not None:
                # calculate the distance between the two obstacles
                dist = np.linalg.norm(ego_state.initial_state.position -
                                      obstacle.occupancy_at_time(time_step).shape.center)

                # add obstacles that are close enough
                if dist < radius:
                    obstacles_within_radius.append(obstacle.obstacle_id)
    if vehicles_in_cone_angle and config:
        obstacles_within_radius = ignore_vehicles_in_cone_angle(time_step, obstacles_within_radius, scenario, ego_state,
                                                                ego_state.obstacle_shape.length,
                                                                config.cone_angle,
                                                                config.cone_safety_dist)
    return obstacles_within_radius


def get_visible_objects(scenario, time_step, ego_state, ego_id=42, sensor_radius=50, occlusions=True, wall_buffer=0.0,
                        vehicles_in_cone_angle=True, config=None):
    """This function simulates a sensor model of a camera/lidar sensor.

    It returns the visible objects and the visible area.

    Arguments:
        scenario {[CommonRoad scenario object]} -- [Commonroad Scenario]
        time_step {[int]} -- [time step for commonroad scenario]
        ego_pos {[list]} -- [list with x and y coordinates of ego position]

    Keyword Arguments:
        sensor_radius {int} -- [description] (default: {50})
        occlusion {bool} -- [True if occlusions by dynamic obstacles should be considered] (default: {True})
        wall_buffer {float} -- [Buffer for visibility around corners in meters] (default: {0.0})


    Returns:
        visible_object_ids [list] -- [list of objects that are visible]
        visible_area [shapely object] -- [area that is visible (for visualization e.g.)]

    """
    ego_pos = ego_state.initial_state.position
    # Create circle from sensor radius
    visible_area = Point(ego_pos).buffer(sensor_radius)

    # Reduce visible area to lanelets
    for idx, lnlet in enumerate(scenario.lanelet_network.lanelets):
        pol_vertices = Polygon(
            np.concatenate((lnlet.right_vertices, lnlet.left_vertices[::-1]))
        )
        if not pol_vertices.is_valid:
            continue

        visible_lnlet = visible_area.intersection(pol_vertices)

        if idx == 0:
            new_vis_area = visible_lnlet
        else:
            new_vis_area = new_vis_area.union(visible_lnlet)

    visible_area = new_vis_area

    # Enlarge visible area by wall buffer
    visible_area = visible_area.buffer(wall_buffer)

    # Substract areas that can not be seen due to geometry
    if visible_area.geom_type == 'MultiPolygon':
        points_vis_area = np.concatenate([np.array(p.exterior.xy).T for p in visible_area.geoms])
    else:
        points_vis_area = np.array(visible_area.exterior.xy).T

    for idx in range(points_vis_area.shape[0] - 1):
        vert_point1 = points_vis_area[idx]
        vert_point2 = points_vis_area[idx + 1]

        pol = _create_polygon_from_vertices(vert_point1, vert_point2, ego_pos)

        if pol.is_valid:
            visible_area = visible_area.difference(pol)

    # if occlusions through dynamic objects should be considered
    if occlusions:

        for obst in scenario.obstacles:
            # check if obstacle is still there
            try:
                if obst.obstacle_role.name == "STATIC":
                    pos = obst.initial_state.position
                    orientation = obst.initial_state.orientation
                else:
                    pos = obst.prediction.trajectory.state_list[time_step].position
                    orientation = obst.prediction.trajectory.state_list[
                        time_step
                    ].orientation
            except IndexError:
                continue

            pos_point = Point(pos)
            # Calculate corner points in world coordinates
            corner_points = _calc_corner_points(pos, orientation, obst.obstacle_shape)
            # Create polygon from corner points
            obst_shape = Polygon(corner_points)

            # check if within sensor radius
            if pos_point.within(visible_area) or obst_shape.intersects(visible_area):
                # Subtract occlusions from dynamic obstacles

                # Identify points for geometric projection
                r1, r2 = _identify_projection_points(corner_points, ego_pos)

                # Create polygon with points far away in the ray direction of ego pos
                r3 = r2 + __unit_vector(r2 - ego_pos) * sensor_radius
                r4 = r1 + __unit_vector(r1 - ego_pos) * sensor_radius

                occlusion = Polygon([r1, r2, r3, r4])

                # Subtract obstacle shape from visible area
                visible_area = visible_area.difference(obst_shape)

                # Subtract occlusion caused by obstacle (everything behind obstacle) from visible area
                if occlusion.is_valid:
                    visible_area = visible_area.difference(occlusion)

    # Get visible objects
    visible_object_ids = []

    for obst in scenario.obstacles:
        # do not consider the ego vehicle
        if obst.obstacle_id != ego_id:
            # check if obstacle is still there
            try:
                if obst.obstacle_role.name == "STATIC":
                    pos = obst.initial_state.position
                    orientation = obst.initial_state.orientation
                else:
                    pos = obst.prediction.trajectory.state_list[time_step].position
                    orientation = obst.prediction.trajectory.state_list[
                        time_step
                    ].orientation

            except IndexError:
                continue

            corner_points = _calc_corner_points(pos, orientation, obst.obstacle_shape)
            obst_shape = Polygon(corner_points)

            if obst_shape.intersects(visible_area):
                visible_object_ids.append(obst.obstacle_id)

    if vehicles_in_cone_angle and config:
        visible_object_ids = ignore_vehicles_in_cone_angle(time_step, visible_object_ids, scenario, ego_state,
                                                           ego_state.obstacle_shape.length,
                                                           config.cone_angle,
                                                           config.cone_safety_dist)

    return visible_object_ids, visible_area


def _calc_corner_points(pos, orientation, obstacle_shape):
    """Calculate corner points of a dynamic obstacles in global coordinate system.

    Arguments:
        pos {[type]} -- [description]
        orientation {[type]} -- [description]
        obstacle_shape {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    corner_points = _rotate_point_by_angle(obstacle_shape.vertices[0:4], orientation)
    corner_points = [p + pos for p in corner_points]
    return np.array(corner_points)


def pi_range(angle):
    """
    Normalize an angle to be within the range [-pi, pi].
    """
    if angle <= -np.pi:
        angle += 2 * np.pi
        return angle
    elif angle > np.pi:
        angle -= 2 * np.pi
        return angle
    return angle


def rotate_glob_loc(global_pos, rot_angle, matrix=True):
    """
    Helper function to rotate positions from global to local coordinates (vehicle coordinates).
    """
    rot_mat = np.array([
        [np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle), np.cos(rot_angle)],
    ])

    return np.dot(rot_mat, global_pos)


def _rotate_point_by_angle(point, angle):
    """Rotate any point by an angle.

    Arguments:
        point {[type]} -- [description]
        angle {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return np.matmul(rotation_matrix, point.transpose()).transpose()


def _identify_projection_points(corner_points, ego_pos):
    """This function identifies the two points of an rectangular objects
    that are the edges from an ego pos point of view.

    Arguments:
        corner_points {[type]} -- [description]
        ego_pos {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    max_angle = 0

    for edge_point1 in corner_points:
        for edge_point2 in corner_points:
            ray1 = edge_point1 - ego_pos
            ray2 = edge_point2 - ego_pos

            angle = _angle_between(ray1, ray2)

            if angle > max_angle:
                max_angle = angle
                ret_edge_point1 = edge_point1
                ret_edge_point2 = edge_point2

    return ret_edge_point1, ret_edge_point2


def _create_polygon_from_vertices(vert_point1, vert_point2, ego_pos):
    """Creates a polygon for the area that is occluded from two vertice points.

    Arguments:
        vert_point1 {[list]} -- [x,y of first point of object]
        vert_point2 {[list]} -- [x,y of second point of object]
        ego_pos {[list]} -- [x,y of ego position]

    Returns:
        pol [Shapely polygon] -- [Represents the occluded area]
    """

    pol_point1 = vert_point1 + 100 * (vert_point1 - ego_pos)
    pol_point2 = vert_point2 + 100 * (vert_point2 - ego_pos)
    pol = Polygon([vert_point1, vert_point2, pol_point2, pol_point1, vert_point1])

    return pol


def __unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def _angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2':"""
    v1_u = __unit_vector(v1)
    v2_u = __unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

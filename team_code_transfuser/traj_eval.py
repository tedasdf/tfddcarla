import numpy as np
from collections import deque
import json

class TrajectoryScoring:
    def __init__(self):

        self.weights = {
            "w_coll": 1.5,
            "w_dev": 5.0,
            "w_dis": 2.5,
            "w_speed": 1.5,
            "w_lat": 4.5,
            "w_lon": 3.0,
            "w_cent": 3.5
            }

        self.speed_history = deque(maxlen=12)  # Store the last average speeds
        self.sample = {}
        self.best_trajectory_queue = deque(maxlen=3) 

    def update_weights(self, response):
        raw_json = response[-1]["message"]["content"]
        weights = json.loads(raw_json)
        self.weights["w_coll"] = weights["Weight_Collision"]
        self.weights["w_dev"] = weights["Weight_Deviation"]
        self.weights["w_dis"] = weights["Weight_Distance"]
        self.weights["w_speed"] = weights["Weight_Speed"]
        self.weights["w_lat"] = weights["Weight_Lat"]
        self.weights["w_lon"] = weights["Weight_Lon"]
        self.weights["w_cent"] = weights["Weight_Cent"]

    def print_values(self):
        print(f"Weight_Collision: {self.weights["w_coll"]}")
        print(f"Weight_Deviation: {self.weights["w_dev"]}")
        print(f"Weight_Distance: {self.weights["w_dis"]}")
        print(f"Weight_Speed: {self.weights["w_speed"]}")
        print(f"Weight_Lat: {self.weights["w_lat"]}")
        print(f"Weight_Lon: {self.weights["w_lon"]}")
        print(f"Weight_Cent: {self.weights["w_cent"]}")
    
    def compute_scores(self, pred_trajectories, target_point):
        """
        Compute predicted trajectories safety and comfort scores
        input: pred_trajectories
        output: scores from each pred_trajectories 
        """
        weights = self.weights

        #target_point = self.sample['target_info'] 
        # If there are best trajectories from the previous frame in the queue, add them to the trajectory list of the current frame.
        if self.best_trajectory_queue:
            previous_best_trajectories = np.array(self.best_trajectory_queue)
            pred_trajectories = np.concatenate([previous_best_trajectories, pred_trajectories], axis=0)
        
        # Calculate average speeds for each trajectory
        average_speeds = self.calculate_average_speeds(pred_trajectories)

        # Update the speed range based on historical average speeds
        if self.speed_history:
            historical_avg_speed = np.mean(self.speed_history)
            self.speed_range = (0.8 * historical_avg_speed, 1.2 * historical_avg_speed)
        else:
            # If there's no history yet, use the current average speeds
            current_avg_speed = np.mean(average_speeds)
            self.speed_range = (0.8 * current_avg_speed, 1.2 * current_avg_speed)

        scores = []
        for i, pred_traj in enumerate(pred_trajectories):
            target_distance = self.calculate_distance_to_target(pred_traj.cpu(), target_point)
            # Calculate the angle deviation of the current trajectory
            angle_deviation_cost = self.calculate_angle_deviation(pred_traj.cpu(), pred_traj[0].cpu(), target_point)
            collision = self.calculate_collisions_with_agents()[i]

            # Calculate speed cost for the current trajectory
            speed_cost = self.calculate_speed_cost([average_speeds[i]], self.speed_range)[0]

            # Calculate dynamics for the trajectory
            long_velocities, lat_velocities, long_accelerations, lat_accelerations, long_jerks, lat_jerks = self.calculate_dynamics(pred_traj.cpu())

            # Calculate comfort costs
            lat_comfort = self.lat_comfort_cost(long_velocities, lat_velocities, long_accelerations, lat_jerks)
            lon_comfort = self.lon_comfort_cost(long_jerks)

            # Calculate centripetal acceleration cost for the current trajectory
            speeds = self.calculate_speeds(np.expand_dims(pred_traj.cpu(), axis=0))[0]
            centripetal_acceleration_cost = self.calculate_centripetal_acceleration_cost(pred_traj.cpu(), speeds)

            # Calculate total score
            total_score = ( weights['w_dis'] * target_distance +
                            weights['w_coll'] * collision +
                            weights['w_speed'] * speed_cost + 
                            weights['w_lat'] * lat_comfort + 
                            weights['w_lon'] * lon_comfort + 
                            weights['w_cent'] * centripetal_acceleration_cost +
                            weights['w_dev'] * angle_deviation_cost
                           )
            
            scores.append(total_score)

        # Update the historical average speed
        self.speed_history.append(np.mean(average_speeds))
        # After the score calculation is finished, find the trajectory with the lowest cost and update the queue.
        min_score_idx = np.argmin(scores[len(self.best_trajectory_queue):])  # 忽略队列中的轨迹
        self.best_trajectory_queue.append(pred_trajectories[min_score_idx])

        return scores
    
    def calculate_angle_deviation(self, trajectory, current_position, target_position):
        target_vector = np.array(target_position) - np.array(current_position)
        angle_deviation_scores = []
        for point in trajectory:
            point_vector = np.array(point) - np.array(current_position)
            dot_product = np.dot(target_vector, point_vector)
            norm_product = np.linalg.norm(target_vector) * np.linalg.norm(point_vector)
            angle_cosine = dot_product / norm_product
            angle_deviation = np.arccos(angle_cosine)  # Value in radians
            angle_deviation_scores.append(angle_deviation)
        return np.mean(angle_deviation_scores)
    
    def calculate_distance_to_target(self, trajectory, target_point):
        return np.linalg.norm(trajectory[-1].cpu() - target_point)

    def check_collision(self, trajectory, other_vehicles_bboxes):
        """
        Check if the trajectory collides with any bounding boxes of other vehicles.
        :param trajectory: Array of shape (T, 2), where T is the number of timesteps.
        :param other_vehicles_bboxes: List of bounding boxes, each defined as [x_center, y_center, width, height, yaw].
        :return: True if collision detected, False otherwise.
        """
        for bbox in other_vehicles_bboxes:
            x_center, y_center, width, height, yaw = bbox
            # Create bounds for the bounding box
            x_min = x_center - width / 2
            x_max = x_center + width / 2
            y_min = y_center - height / 2
            y_max = y_center + height / 2

            # Check each point in the trajectory
            for point in trajectory:
                x, y = point
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True  # Collision detected
        return False  # No collision detected
    
    def calculate_collisions_with_agents(self):
        trajectories = self.sample['pred_ego_fut_trajs']
        agent_boxes = self.sample['gt_attr_labels']
        collision_scores = []
        for traj in trajectories:
            collision = 0
            for agent in agent_boxes:
                # Convert agent box format to [x_center, y_center, width, height, yaw]
                bbox = [agent[0], agent[1], agent[3], agent[4], agent[2]]
                if self.check_collision(traj, [bbox]):  # Check collision with each agent
                    collision = 1
                    break
            collision_scores.append(collision)
        return collision_scores

    def calculate_average_speeds(self, trajectories):
        """
        Calculate average speeds for each trajectory.
        :param trajectories: Array of shape (num_trajectories, num_points, 2)
        :return: Array of average speeds of shape (num_trajectories,)
        """
        speeds = self.calculate_speeds(trajectories.cpu())
        average_speeds = np.mean(speeds, axis=1)
        return average_speeds

    def calculate_speeds(self, trajectories):
        """
        Calculate speeds for each point in each trajectory.
        :param trajectories: Array of shape (num_trajectories, num_points, 2)
        :return: Array of speeds of shape (num_trajectories, num_points-1)
        """
        speeds = np.linalg.norm(np.diff(trajectories, axis=1), axis=2)
        return speeds

    def calculate_dynamics(self, trajectory):
        """
        Calculate longitudinal and lateral velocities, accelerations, and jerks for a trajectory.
        """
        dt = 0.1  # assuming trajectory points are at 0.1 second intervals
        velocities = np.diff(trajectory, axis=0) / dt
        long_velocities = velocities[:, 0]  # Assuming x is the longitudinal direction
        lat_velocities = velocities[:, 1]  # Assuming y is the lateral direction

        long_accelerations = np.diff(long_velocities) / dt
        lat_accelerations = np.diff(lat_velocities) / dt

        long_jerks = np.diff(long_accelerations) / dt
        lat_jerks = np.diff(lat_accelerations) / dt

        return long_velocities, lat_velocities, long_accelerations, lat_accelerations, long_jerks, lat_jerks
    
    def lat_comfort_cost(self, long_velocities, lat_velocities, long_accelerations, lat_jerks):
        max_cost = 0.0
        for i in range(len(long_velocities)):
            s_dot = long_velocities[i]
            s_dotdot = long_accelerations[i] if i < len(long_accelerations) else 0.0
            l_prime = lat_velocities[i]
            l_primeprime = lat_jerks[i] if i < len(lat_jerks) else 0.0
            cost = l_primeprime * s_dot * s_dot + l_prime * s_dotdot
            max_cost = max(max_cost, abs(cost))
        return max_cost

    def lon_comfort_cost(self, long_jerks):
        cost_sqr_sum = 0.0
        cost_abs_sum = 0.0
        longitudinal_jerk_upper_bound = 5.0  
        numerical_epsilon = 1e-6  
        for jerk in long_jerks:
            cost = jerk / longitudinal_jerk_upper_bound
            cost_sqr_sum += cost * cost
            cost_abs_sum += abs(cost)
        return cost_sqr_sum / (cost_abs_sum + numerical_epsilon)

    def calculate_speed_cost(self, average_speeds, speed_range):
        # acording to the limitation of scenario setting speed or some shit
        if self.driving_scene == 'city':
            speed_limit = self.default_city_road_speed_limit
        elif self.driving_scene == 'highway':
            speed_limit = self.default_highway_speed_limit
        else:
            raise ValueError("Invalid driving scene type. Must be 'city' or 'highway'.")

        speed_costs = []
        for speed in average_speeds:
            # if the speed exceed limit set the cost to inf
            if speed > speed_limit:
                speed_cost = float('inf')
            else:
                # within the speed range compute cost 
                if self.driving_style == 'aggressive':
                    # aggressive driving style
                    if speed < speed_range[0]:
                        speed_cost = (speed_range[0] - speed) / speed_range[0]
                    else:
                        speed_cost = 0
                elif self.driving_style == 'conservative':
                    if speed > speed_range[1]:
                        speed_cost = (speed - speed_range[1]) / speed_range[1]
                    else:
                        speed_cost = 0
            speed_costs.append(speed_cost)

        return speed_costs

    def calculate_centripetal_acceleration_cost(self, trajectory, speeds):
        """
        Calculate the centripetal acceleration cost for a single trajectory.
        :param trajectory: Array of shape (num_points, 2) containing a single trajectory.
        :param speeds: Array of speeds of shape (num_points-1,) for the trajectory.
        :return: Centripetal acceleration cost for the trajectory.
        """
        centripetal_acc_sum = 0.0
        centripetal_acc_sqr_sum = 0.0
        numerical_epsilon = 1e-6  # Avoid division by zero

        # Calculate curvature
        curvatures = [self.calculate_curvature(trajectory[i - 1], trajectory[i], trajectory[i + 1])
                      for i in range(1, len(trajectory) - 1)]

        # Calculate centripetal acceleration and cost
        for i, curvature in enumerate(curvatures):
            centripetal_acc = speeds[i] ** 2 * curvature
            centripetal_acc_sum += np.fabs(centripetal_acc)
            centripetal_acc_sqr_sum += centripetal_acc ** 2

        cost = centripetal_acc_sqr_sum / (centripetal_acc_sum + numerical_epsilon)
        return cost

    def calculate_curvature(self, p1, p2, p3):
        """
        Calculate the curvature given three consecutive points on the trajectory.
        :param p1, p2, p3: Consecutive points on the trajectory.
        :return: Curvature.
        """

        k = np.linalg.norm(np.cross(p2 - p1, p3 - p2)) / np.linalg.norm(p2 - p1)**2
        return k


    
   
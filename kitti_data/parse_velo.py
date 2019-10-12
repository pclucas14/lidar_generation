import pdb
import numpy as np

def get_quadrant(point):
    if point[0] >= 0. and point[1] >= 0. :
        return 0
    elif point[0] <= 0. and point[1] >= 0. : 
        return 1
    elif point[0] <= 0. and point[1] <= 0. : 
        return 2
    elif point[0] >= 0. and point[1] <= 0. : 
        return 3
    else :
        raise Exception('invalid input %s', point) 


def passed_origin(x_t, x_t1):
    if get_quadrant(x_t1) == 3 and get_quadrant(x_t) == 0: 
        return True
    else : 
        return False


def fit_quadrant(points, quadrant, desired_amt):
    points = np.asarray(points)
    slots = []
    slot_size = np.pi / (2 * desired_amt)
    for i in range(desired_amt) : slots.append([])
    if quadrant == 0: 
        points = points[::-1]
    elif quadrant == 1 : 
        points[:, 0] = - points[:, 0]
    elif quadrant == 2 :
        points = points[::-1] 
        points[:, 0] = - points[:, 0]
        points[:, 1] = - points[:, 1]
    elif quadrant == 3 : 
        points[:, 1] = - points[:, 1]

    # import pdb; pdb.set_trace()
    for point in points : 
        angle = np.arctan(point[1] / point[0])
        index = min(int(angle / slot_size), desired_amt - 1)
        slots[index].append(point)

    for i in range(len(slots)):
        if len(slots[i]) == 0 : 
            slots[i] = np.array([0., 0., 0., 0.])
        else :
            full_slot = np.asarray(slots[i])
            slots[i] = full_slot.mean(axis=0)

    points = np.asarray(slots)
    if quadrant == 0: 
        points = points[::-1]
    elif quadrant == 1 : 
        points[:, 0] = - points[:, 0]
    elif quadrant == 2 : 
        points = points[::-1]
        points[:, 0] = - points[:, 0]
        points[:, 1] = - points[:, 1]
    elif quadrant == 3 : 
        points[:, 1] = - points[:, 1]

    return points

def parse_velo(velo):
    # points closer to the origin (0,0,0) are at the end of the point cloud.
    # invert the point cloud such that we begin near the origin. 
    
    # returns: a H x 4 x ? array, split into quadrants
    velo = velo[::-1]
    lines = []
    current_point = velo[0]
    current_quadrant = get_quadrant(current_point)
    current_line = [[], [], [], []]
    quadrant_switches = 0
    for point in velo :
        point_quadrant = get_quadrant(point)
        
        if passed_origin(current_point, point):
            lines.append(current_line)
            current_line = [[], [], [], []]

        current_line[point_quadrant].append(point)
        current_quadrant = point_quadrant
        current_point = point

    return lines

def process_velo(velo, points_per_layer, stop=False):
    lines = parse_velo(velo)
    inverse = quad_to_pc_inv(lines)
    lines = lines[2:-1]
    if len(lines) != 60 : raise Exception('invalid nb un of lines')
    out_tensor = np.zeros((60, points_per_layer, 4))
    if stop:
        import pdb; pdb.set_trace()
        x = 1
    for j in range(len(lines)):
        line = lines[j]
        out_line = np.zeros((points_per_layer, 4))
        for i in range(len(line)):
            gridded = fit_quadrant(line[i], i, points_per_layer / 4)
            out_tensor[j][i*points_per_layer/4:(i+1)*points_per_layer/4, :] = gridded[::-1]

    return out_tensor, inverse


def quad_to_pc_inv(lines, th=3.):
    # lines is a 63 x 4 array, where each slot has an array of 4d/3d points
    # goal : get an array of points that fills empty spaces
    points = []
    for i in range(len(lines)) :
        line = lines[i] 
        distance = []
        for quad in line : 
            for point in quad : 
                x, y, z = point[:3]
                distance.append(x**2 + y**2)
        distance = np.array(distance)
        std = distance.std()
        sorted_indices = np.argsort(distance)
        median_index = sorted_indices[int(sorted_indices.shape[0]*0.95)]
        median = distance[median_index]

        for quad in line : 
            for point in quad : 
                x, y, z = point[:3]
                dist = x ** 2 + y ** 2 
                if dist < median and (median/dist-1.) > th:#*std : 
                    # blocked point --> scale to get real pt
                    scale = np.sqrt(median / dist)
                    scaled = scale * point
                    points.append(scaled)


    return np.array(points)

if __name__ == '__main__':
    x = np.random.normal(size=(1000, 3))
    polar = process_velo(x, 512)


import numpy as np

class WidthComputer:
    def __init__(self):
        pass

    # def calculate_center_point(self, point1, point2):
    #     """
    #     计算两点的中心点
    #     """
    #     return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

    # def find_nearest_points(self, points, reference_point, k=5):
    #     """
    #     找到与给定点距离最近的k个点，并返回它们的索引
    #     """
    #     distances = [np.linalg.norm(np.array(point) - np.array(reference_point)) for point in points]
    #     nearest_indices = np.argsort(distances)[:k]
    #     return nearest_indices

    # def find_shortest_distance_between_points(self, points):
    #     """
    #     找到点列表中两点之间的最短距离，并返回这两个点
    #     """
    #     min_distance = float('inf')
    #     closest_pair = None
    #     for i in range(len(points)):
    #         for j in range(i + 1, len(points)):
    #             distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 closest_pair = [points[i], points[j]]
    #     return closest_pair, min_distance
   
    def vectors_method(self, boundary_points, center_point):
        # print(len(boundary_points))
        # 计算每个边界点到中心点的向量
        vectors_to_center = boundary_points - center_point

        # 计算每个边界点到中心点连线的垂直距离
        vertical_distances = np.abs(np.cross(vectors_to_center, np.array([1, 0])))

        # 找到距离中心点最远和最近的边界点的索引
        farthest_index = np.argmax(vertical_distances)
        # print(farthest_index)
        nearest_index = np.argmin(vertical_distances)

        # 使用索引找到对应的边界点
        farthest_point = boundary_points[farthest_index]
        nearest_point = boundary_points[nearest_index]

        # 计算最远和最近边界点之间的距离，即为物体的宽度
        width0 = np.linalg.norm(farthest_point - nearest_point)

        print("Width of the object:", width0)

        return width0

    def near_far_method(self, boundary_points , center_point):

        # 计算每个边界点到中心点的距离
        distances_to_center = np.linalg.norm(boundary_points - center_point, axis=1)
        # print(distances_to_center)

        # 找到距离中心点最远和最近的边界点的索引
        farthest_index = np.argmax(distances_to_center)
        # print(farthest_index)
        nearest_index = np.argmin(distances_to_center)

        # 使用索引找到对应的边界点
        farthest_point = boundary_points[farthest_index]
        nearest_point = boundary_points[nearest_index]

        # 计算最远和最近边界点之间的距离，即为物体的宽度
        width1 = np.linalg.norm(farthest_point - nearest_point)

        print("Width of the object:", width1)

        return width1


    def box_method(self, boundary_points):
        # 计算多边形的外接矩形
        min_x = np.min(boundary_points[:, 0])
        max_x = np.max(boundary_points[:, 0])
        min_y = np.min(boundary_points[:, 1])
        max_y = np.max(boundary_points[:, 1])

        # 计算矩形的宽度
        width2 = min(max_x - min_x, max_y - min_y)

        print("Width of the object:", width2)
        return width2

if __name__ == "__main__":
    point_processor = WidthComputer()

    boundary_points_ = np.array([[307, 244],
                                [306, 245],
                                [305, 245],
                                [302, 248],
                                [289, 248],
                                [288, 249],
                                [287, 249],
                                [285, 251],
                                [283, 251],
                                [282, 252],
                                [273, 252],
                                [272, 253],
                                [272, 266],
                                [273, 267],
                                [277, 267],
                                [280, 264],
                                [281, 264],
                                [282, 263],
                                [298, 263],
                                [299, 262],
                                [301, 262],
                                [303, 260],
                                [305, 260],
                                [306, 259],
                                [324, 259],
                                [325, 258],
                                [326, 258],
                                [327, 257],
                                [327, 256],
                                [328, 255],
                                [328, 248],
                                [327, 247],
                                [327, 245],
                                [326, 245],
                                [325, 244]], dtype=np.float32)
    center_point_ = np.array([298, 254])  # 中心点
    width0 = WidthComputer().vectors_method(boundary_points=boundary_points_, center_point=center_point_)
    width1 = WidthComputer().near_far_method(boundary_points_, center_point_)
    width2 = WidthComputer().box_method(boundary_points_)
    width = [width0, width1, width2]
    if np.round(np.mean(width)) % 10 >= 5:
        width_ = np.round(np.mean(width) + (10 - np.round(np.mean(width)) % 10)) * 0.001

    else:
        width_ = np.round(np.mean(width) - np.round(np.mean(width)) % 10) * 0.001
        print(width_)

    # threshold_distance = 3  # 设定距离阈值为10

    # # 找到距离最近的5个点
    # nearest_indices = point_processor.find_nearest_points(points, reference_point, k=5)
    # print(nearest_indices)
    # nearest_points = [points[i] for i in nearest_indices]

    # # 过滤掉与相邻点距离小于等于阈值的点
    # filtered_points = [nearest_points[0]]
    # for i in range(1, len(nearest_points)):
    #     if np.linalg.norm(np.array(nearest_points[i]) - np.array(nearest_points[i-1])) > threshold_distance:
    #         filtered_points.append(nearest_points[i])

    # # 打印最近五个点之间的两两距离以及它们与参考点的距离
    # print("Distances between Nearest Points:")
    # for i in range(len(filtered_points)):
    #     for j in range(i + 1, len(filtered_points)):
    #         distance_ij = np.linalg.norm(np.array(filtered_points[i]) - np.array(filtered_points[j]))
    #         distance_to_reference_i = np.linalg.norm(np.array(filtered_points[i]) - np.array(reference_point))
    #         distance_to_reference_j = np.linalg.norm(np.array(filtered_points[j]) - np.array(reference_point))
    #         # print(f"Point {i+1} to Point {j+1} Distance:", distance_ij)
    #         # print(f"Point {i+1} to Reference Point Distance:", distance_to_reference_i)
    #         # print(f"Point {j+1} to Reference Point Distance:", distance_to_reference_j)
    #         # print()

    # # 计算最近点对应的最短距离
    # closest_pair, min_distance = point_processor.find_shortest_distance_between_points(filtered_points)
    # grasp_dist = min_distance * 0.001
    # print(grasp_dist)

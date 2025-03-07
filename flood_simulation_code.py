# flood_simulation_code.py


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rasterio
import math
import numpy.ma as ma
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import csv
import matplotlib
from shapely import LineString, MultiLineString
from shapely.geometry import Point
from shapely.geometry import box
import numpy as np
from scipy.ndimage import binary_dilation
import geopandas as gpd
import time
from rasterio.warp import calculate_default_transform, reproject
from pyproj import CRS

start = time.perf_counter()

# 设置全局字体为宋体
matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 宋体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

matplotlib.use('Agg')

# 全局状态记录
road_previous_state = {}
base_station_previous_state = {}
power_station_previous_state = {}

# 创建标注字典
marked_power_station_texts = {}
marked_base_station_texts = {}
previous_risk_value = {}

# 保存影响关系的直线连接（起点和终点坐标）
influence_connections = []

# 字典记录设施的破坏帧数
road_damage_durations = {}  # 用于记录每条道路的破坏时长
base_station_damage_durations = {}
power_station_damage_durations = {}
building_damage_durations = {}
# 记录连接数量
base_station_counts = {}
power_station_counts = {}
road_power_station_counts = {}
# 链风险值
p_b_risk_values = {}
b_b_risk_values = {}
p_r_risk_values = {}
# 记录发生影响的帧数
road_influence_durations = {}  # 用于记录每条道路的破坏时长
base_station_influence_durations = {}

# 损失风险变量
total_road_risk = 0
total_power_station_risk = 0
total_base_station_risk = 0
unit_damage_loss_road = 0
unit_damage_loss_power_station = 0
unit_damage_loss_base_station = 0
total_connection_risk = 0
total_building_risk = 0
unit_damage_loss_building = 0
total_building_damaged_loss = 0
total_building_influenced_loss = 0
total_road_damaged_loss = 0
total_road_influenced_loss = 0
total_power_station_damaged_loss = 0
total_base_station_damaged_loss = 0
total_base_station_influenced_loss = 0



# 主函数：洪水模拟
def flood_simulation(buildings_file, roads_file, base_stations_file, power_stations_file, output_folder,
                     output_shp_folder,
                     water_depth_folder, interval=300):
                       
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取矢量数据
    building_gdf = buildings_file
    road_gdf = roads_file
    base_station_gdf = base_stations_file
    power_station_gdf = power_stations_file
    # 变换投影
    building_gdf = building_gdf.to_crs(epsg=32650)
    road_gdf = road_gdf.to_crs(epsg=32650)
    base_station_gdf = base_station_gdf.to_crs(epsg=32650)
    power_station_gdf = power_station_gdf.to_crs(epsg=32650)
    building_gdf['building_id'] = range(len(building_gdf))
    road_gdf['road_id'] = range(len(road_gdf))
    base_station_gdf['base_station_id'] = range(len(base_station_gdf))
    power_station_gdf['power_station_id'] = range(len(power_station_gdf))

    # 创建CSV文件并写入标题行
    # 定义保存文件的文件夹和文件路径
    # output_folder = "output_folder"
    output_file = os.path.join(output_folder, "risk_loss_per_frame.csv")
    if not os.path.exists(output_file):
        with open(output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["frame",
                 #"building_damage_area", "building_influenced_area", "building_normal_area", " power_station_damage_count", "tpower_station_influenced_count",
                 #"power_station_normal_count","base_station_damage_count", "base_station_influenced_count", "base_station_normal_count"
                 "road_damage_length", "road_influenced_length", "road_normal_length", "road_damage_count", "road_influenced_count", "road_normal_count"
                 ])

    # 在模拟每一帧中记录数据
    def record_frame_data(frame ,
                          #building_damage_area ,building_influenced_area ,building_normal_area,
                          #power_station_damage_count ,power_station_influenced_count ,power_station_normal_count ,
                          #base_station_damage_count ,base_station_influenced_count , base_station_normal_count,
                          road_damage_length,road_influenced_length,road_normal_length
                          ,road_damage_count,road_influenced_count,road_normal_count
                          ):
        with open(output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [frame ,
                 # building_damage_area ,building_influenced_area ,building_normal_area,
                 # power_station_damage_count ,power_station_influenced_count ,power_station_normal_count ,
                 # base_station_damage_count ,base_station_influenced_count , base_station_normal_count,
                 road_damage_length, road_influenced_length, road_normal_length
                 , road_damage_count, road_influenced_count, road_normal_count
                 ])

    def save_influence_lines(frame, influence_connections):
        lines = []
        start_coords = []  # 用于保存起点坐标
        end_coords = []  # 用于保存终点坐标

        for connection in influence_connections:
            # print(connection)

            # 解析坐标
            start_x, end_x = (connection[0][0], connection[0][1])
            start_y, end_y = (connection[1][0], connection[1][1])

            # 创建LineString对象
            line = LineString([(start_x, start_y), (end_x, end_y)])

            # 将线条和坐标信息添加到列表
            lines.append(line)
            start_coords.append((start_x, start_y))
            end_coords.append((end_x, end_y))

        if lines:  # 检查是否有有效的线条
            lines_gdf = gpd.GeoDataFrame({
                'geometry': lines,
                'start_x': [coord[0] for coord in start_coords],
                'start_y': [coord[1] for coord in start_coords],
                'end_x': [coord[0] for coord in end_coords],
                'end_y': [coord[1] for coord in end_coords]
            }, crs="EPSG:32650")  # 使用合适的投影坐标系
            lines_filename = os.path.join("output_shp", f"influence_lines_frame_{frame:03d}.shp")

            lines_gdf.to_file(lines_filename, driver="ESRI Shapefile", encoding="utf-8")
        else:
            print(f"No influence lines to save for frame {frame}.")

    water_depth_file = os.path.join(water_depth_folder, f'water_depth_frame_000.tif')
    with rasterio.open(water_depth_file) as water_depth_raster:
        water_depth = water_depth_raster.read(1)
        transform = water_depth_raster.transform
        grid_size = water_depth.shape

    def sum_of_connection(x, n):
        total_sum = 0
        for i in range(1, n + 1):
            total_sum += x ** (1 / i)
        return total_sum

    # 计算坐标到栅格索引的转换
    def coord_to_index(x, y, transform):
        col, row = ~transform * (x, y)
        return int(row), int(col)

    # 初始化设施状态
    def initialize_facility_status(gdf: object) -> object:
        gdf['status'] = 'Normal'
        return gdf

    building_gdf = initialize_facility_status(building_gdf)
    road_gdf = initialize_facility_status(road_gdf)
    base_station_gdf = initialize_facility_status(base_station_gdf)
    power_station_gdf = initialize_facility_status(power_station_gdf)

    # 设置设施的权重（用于灾害传播）
    facility_weights = {
        'road': 2.0,
        'base_station': 1.0,
        'power_station': 5.0
    }

    # 计算破坏概率函数（电站）
    def calculate_power_station_damage_probability(depth):
        if depth <= 0.3:
            return 0
        elif 0.3 <= depth < 0.5:
            return 0.25
        elif 0.5 <= depth < 1.8:
            return 0.5
        elif 1.8 <= depth < 2.0:
            return 0.75
        else:
            return 1

    # 计算破坏概率函数（基站）
    def calculate_base_station_damage_probability(time_of_damaged):
        if time_of_damaged <= 10 * 60:
            return 0.05
        elif 10 * 60 < time_of_damaged <= 30 * 60:
            return 0.25
        elif 30 * 60 < time_of_damaged <= 60 * 60:
            return 0.5
        elif 60 * 60 < time_of_damaged <= 180 * 60:
            return 0.75
        else:
            return 1

    # 计算破坏概率函数（道路）
    def calculate_road_damage_probability(depth):
        if depth is not None:
            if depth >= 0.3:
                return 1
            else:
                return 1 - 1 / 2 * math.tanh((-depth * 10 + 15) / 4) + 1 / 2  # 水深-破坏概率曲线
        else:
            return 0

    # 检查道路被淹没情况，并返回最大水深值
    def is_road_flooded(road, transform, water_depth, num_points=10):
        line = road.geometry
        depths = []  # 存储所有采样点的水深值
        points = [line.interpolate(float(i) / num_points, normalized=True) for i in range(num_points)]
        for point in points:
            row, col = coord_to_index(point.x, point.y, transform)
            if 0 <= row < water_depth.shape[0] and 0 <= col < water_depth.shape[1]:
                depth = water_depth[row, col]
                if depth > 0:
                    depths.append(depth)
        return max(depths) if depths else None  # 返回最大水深或None

    def center_point_on_line(line):
        if line.geom_type == 'LineString':
            line = line
        elif line.geom_type == 'MultiLineString':
            line = max(line.geoms, key=lambda l: l.length)  # 最长线
        else:
            print(f"Unexpected geometry type {line.geom_type}")
        # 计算长度
        total_length = line.length
        # 寻找线上中点
        midpoint = line.interpolate(total_length / 2)
        return midpoint.x, midpoint.y

    # 初始化绘图区域
    fig, ax = plt.subplots(figsize=(20, 20))  # 创建figure和子图对象ax

    # 更新洪水扩散和设施状态的函数
    def update_flood(frame):
        global water_depth, base_station_previous_state, power_station_previous_state, influence_connections, total_building_risk, unit_damage_loss_building
        global total_power_station_risk, total_road_risk, total_base_station_risk, road_power_station_counts
        global current_road_loss, unit_damage_loss_road, unit_damage_loss_power_station, unit_damage_loss_base_station
        global base_station_influence_durations, road_influence_durations, total_connection_risk, base_station_counts, power_station_counts, marked_power_station_texts, marked_base_station_texts
        global p_r_risk_values, p_b_risk_values, b_b_risk_values, previous_risk_value, road_previous_state, power_station_damage_durations, base_station_damage_durations
        global total_building_influenced_loss, total_power_station_damaged_loss, total_building_damaged_loss, total_road_damaged_loss, total_road_influenced_loss, total_base_station_damaged_loss, total_base_station_influenced_loss



        if frame == 0 and hasattr(update_flood, 'init_called'):
            return
        if frame == 0:
            update_flood.init_called = True
        frame_count = frame + 1

        risk = 0

        current_road_loss=0
        current_frame_building_loss=0
        current_frame_power_station_loss=0
        current_frame_base_station_loss=0
        total_road_risk = 0
        total_building_risk = 0
        total_power_station_risk = 0
        total_base_station_risk = 0

        # 影响传播延时
        interval_base_station = 0
        interval_road = 0

        # 参数设置
        threshold = 0  # 水深阈值（单位：米）
        dilation_distance = 50  # 扩展距离（单位：米）

        # 导入当前帧的水深数据
        water_depth_file = os.path.join(water_depth_folder, f'water_depth_frame_{frame:03d}.tif')
        with rasterio.open(water_depth_file) as water_depth_raster:
            water_depth = water_depth_raster.read(1)
            transform = water_depth_raster.transform  # 获取地理坐标转换
            crs = water_depth_raster.crs  # 获取投影信息
            pixel_size = water_depth_raster.res[0]  # 栅格分辨率（假定为方形像素）
            grid_size = water_depth.shape

        # 确定被淹没区域（超过阈值的区域）
        flooded_area = water_depth > threshold  # 超过阈值的区域为 True

        # 扩展淹没范围
        dilation_pixels = int(dilation_distance / pixel_size)  # 转换为像素数
        extended_area = binary_dilation(flooded_area, iterations=dilation_pixels)

        #  将扩展后的区域转为 GeoDataFrame
        rows, cols = np.where(extended_area)  # 获取被扩展区域的像素索引
        flood_bounds = [box(
            transform[2] + col * transform[0],  # 左
            transform[5] + row * transform[4],  # 下
            transform[2] + (col + 1) * transform[0],  # 右
            transform[5] + (row + 1) * transform[4]  # 上
        ) for row, col in zip(rows, cols)]  # 转换为几何对象
        flood_gdf = gpd.GeoDataFrame(geometry=flood_bounds, crs=crs)
        flood_gdf = flood_gdf.to_crs(epsg=32650)
        # 清除之前的绘图
        ax.clear()


        # 仅显示淹没范围
        masked_water = ma.masked_where(water_depth == 0, water_depth)
        extent = [water_depth_raster.bounds.left, water_depth_raster.bounds.right, water_depth_raster.bounds.bottom,
                  water_depth_raster.bounds.top]
        ax.imshow(masked_water, cmap='Blues', alpha=0.5, extent=extent, zorder=0)

        # 假设的初始单位损失
        initial_unit_damage_loss_building = 1.64E-06
        initial_unit_damage_loss_road = 0.02483
        initial_unit_damage_loss_base_station = 1.79895
        initial_unit_damage_loss_power_station = 1.64E-06

        # 假设每帧的时间（5分钟）
        frame_duration = interval

        # 6. 判断建筑是否受到影响
        affected_buildings = gpd.sjoin(building_gdf, flood_gdf, how="inner", op="intersects")
        # 判断破坏基站
        affected_base_station = gpd.sjoin(base_station_gdf, flood_gdf, how="inner", op="intersects")

        # 7. 更新建筑状态为 "受影响"
        building_gdf["status"] = "Normal"  # 初始化状态为 "未受影响"
        influenced_indices = affected_buildings.index  # 获取受影响建筑的索引
        building_gdf.loc[influenced_indices, "status"] = "Influenced"  # 更新受影响建筑的状态

        # print('wai:',total_building_risk)

        building_damage_area = 0
        building_influenced_area = 0
        building_normal_area = 0

        # 更新并绘制建筑状态
        for idx, building in building_gdf.iterrows():
            # print(total_building_risk)
            x, y = building.geometry.centroid.x, building.geometry.centroid.y
            row, col = coord_to_index(x, y, transform)
            # if building['status'] == 'Normal':
            if 0 <= row < grid_size[0] and 0 <= col < grid_size[1]:
                depth = water_depth[row, col]
                if depth > 0:
                    building_gdf.at[idx, 'status'] = 'Damaged'
                    if idx not in building_damage_durations:
                        building_damage_durations[idx] = frame_count


            if building_gdf.at[idx, 'status'] == 'Damaged':
                area = building.geometry.area
                damage_duration = frame_duration
                unit_damage_loss_building = initial_unit_damage_loss_building * damage_duration * area
                total_building_damaged_loss += unit_damage_loss_building
                # print('frame:', frame, 'current_frame_building_loss1:', current_frame_building_loss)

                damage_duration = frame_duration * (frame_count - building_damage_durations[idx] + 1)
                unit_damage_loss_building2 = initial_unit_damage_loss_building * damage_duration * area
                building_gdf.at[idx, 'loss'] = unit_damage_loss_building2  # shp文件属性表添加风险字段
                probability = 1
                risk = probability * unit_damage_loss_building2 / area
                building_gdf.at[idx, 'risk'] = risk  # shp文件属性表添加风险字段
                total_building_risk += risk
                # print('frame:', frame, 'total_building_risk1:', total_building_risk)

            if building_gdf.at[idx, 'status'] == 'Influenced':
                if idx not in building_damage_durations:
                    building_damage_durations[idx] = frame_count

                building_area = building.geometry.area

                damage_duration = frame_duration
                unit_damage_loss_building = initial_unit_damage_loss_building * damage_duration * building_area
                total_building_influenced_loss += unit_damage_loss_building
                # print('frame:', frame, 'current_frame_building_loss2:', current_frame_building_loss)

                damage_duration = frame_duration * (frame_count - building_damage_durations[idx] + 1)
                unit_damage_loss_building2 = initial_unit_damage_loss_building * damage_duration * building_area
                probability = 1
                risk = probability * unit_damage_loss_building2 / building_area
                building_gdf.at[idx, 'risk'] = risk
                total_building_risk += risk

            # 根据建筑状态设置颜色
            if building_gdf.at[idx, 'status'] == 'Damaged':
                building_color = 'red'
                building_damage_area+=building.geometry.area
            elif building_gdf.at[idx, 'status'] == 'Influenced':
                building_color = 'gold'
                building_influenced_area +=building.geometry.area
            else:
                building_color = 'LightGrey'
                building_normal_area +=building.geometry.area

            if building.geometry.geom_type == 'Polygon':
                ax.fill(*building.geometry.exterior.xy, color=building_color)
            elif building.geometry.geom_type == 'MultiPolygon':
                for polygon in building.geometry.geoms:
                    ax.fill(*polygon.exterior.xy, color=building_color, zorder=2)

        # 绘制所有已经记录的影响关系直线（保留之前帧的影响线）
        for connection in influence_connections:
            ax.plot(connection[0], connection[1], color='purple', linestyle='--', linewidth=1)

        # 更新电站状态并保留上一帧的状态
        for idx, station in power_station_gdf.iterrows():
            x, y = station.geometry.x, station.geometry.y
            row, col = coord_to_index(x, y, transform)

            # 保持上一帧的状态
            previous_status = power_station_previous_state.get(idx, 'Normal')

            if previous_status == 'Damaged':
                power_station_gdf.at[idx, 'status'] = 'Damaged'
            elif station['status'] == 'Normal' and 0 <= row < grid_size[0] and 0 <= col < grid_size[1]:
                depth = water_depth[row, col]
                if depth > 0.3:
                    power_station_gdf.at[idx, 'status'] = 'Damaged'
                    if idx not in power_station_damage_durations:
                        power_station_damage_durations[idx] = frame_count

            # 绘制电站状态（修改）
            # 初始化计数变量
            power_station_damage_count = 0
            power_station_influenced_count = 0
            power_station_normal_count=0
            # 遍历电站
            for idx, power_station in power_station_gdf.iterrows():
                # 获取电站位置
                x, y = power_station.geometry.x, power_station.geometry.y
                # 根据电站状态设置颜色
                if power_station_gdf.at[idx, 'status'] == 'Damaged':
                    power_color = 'red'
                    power_station_damage_count += 1  # 统计破坏电站数量
                else:
                    power_color = 'blue'
                    power_station_normal_count += 1  # 统计正常电站数量
                # 绘制电站
                ax.scatter(x, y, color=power_color, s=25, marker='s', zorder=4)

            # 计算损失和风险
            if power_station_gdf.at[idx, 'status'] == 'Damaged':
                if idx not in power_station_damage_durations:
                    power_station_damage_durations[idx] = frame_count

                damage_duration = frame_duration
                unit_damage_loss_power_station = initial_unit_damage_loss_power_station * damage_duration
                total_power_station_damaged_loss += unit_damage_loss_power_station


                damage_duration = frame_duration * (frame_count - power_station_damage_durations[idx] + 1)
                unit_damage_loss_power_station2 = initial_unit_damage_loss_power_station * damage_duration

                depth = water_depth[row, col]
                probability = calculate_power_station_damage_probability(depth)
                # plt.text(x, y, f'{probability:.2f}', fontsize=3, color='black')
                risk = probability * unit_damage_loss_power_station2
                power_station_gdf.at[idx, 'risk'] = risk
                total_power_station_risk += risk

            # 更新上一帧状态
            power_station_previous_state[idx] = power_station_gdf.at[idx, 'status']

        road_damage_length = 0
        road_influenced_length = 0
        road_normal_length = 0
        road_damage_count = 0
        road_influenced_count = 0
        road_normal_count = 0
        # 更新并绘制道路状态
        for idx, road in road_gdf.iterrows():
            # 保持上一帧的状态
            previous_status = road_previous_state.get(idx, 'Normal')

            # 如果上一帧已受损则保留该状态
            if previous_status in ['Damaged', 'Influenced']:
                road_gdf.at[idx, 'status'] = previous_status

            max_depth = is_road_flooded(road, transform, water_depth)  # 获取最大水深值
            road_geometr = road['geometry']
            center_x, center_y = center_point_on_line(road_geometr)
            if max_depth is not None:
                road_gdf.at[idx, 'status'] = 'Damaged'

            # 选择颜色
            if road_gdf.at[idx, 'status'] == 'Damaged':
                max_depth = is_road_flooded(road, transform, water_depth)
                if idx not in road_damage_durations:
                    road_damage_durations[idx] = frame_count


                damage_duration = frame_duration
                road_length = road.geometry.length
                unit_damage_loss_road = initial_unit_damage_loss_road * (
                        damage_duration + road_length / 16.67)  # 该时刻损失
                total_road_damaged_loss += unit_damage_loss_road

                # color = 'Maroon'
                probability = calculate_road_damage_probability(max_depth)  # 基于最大水深计算破坏概率
                # 在道路中间绘制概率
                mid_x, mid_y = road.geometry.centroid.x, road.geometry.centroid.y


                # 计算风险
                risk = probability * unit_damage_loss_road

                total_road_risk += risk  # 加入当前帧风险

                # 计算破坏持续时间并更新损失
                damage_duration = frame_duration * (frame_count - road_damage_durations[idx] + 1)
                road_length = road.geometry.length
                unit_damage_loss_road2 = initial_unit_damage_loss_road * (
                        damage_duration +road_length / 16.67)  # 该时刻损失
                risk = probability * unit_damage_loss_road2
                road_gdf.at[idx, 'risk'] = risk

            elif road_gdf.at[idx, 'status'] == 'Normal':
                # 遍历所有受损的电站和道路
                for ps_idx, power_station in power_station_gdf.iterrows():
                    if power_station_gdf.at[ps_idx, 'status'] == 'Damaged':
                        p_x, p_y = power_station.geometry.x, power_station.geometry.y
                        p_row, p_col = coord_to_index(p_x, p_y, transform)
                        depth = water_depth[p_row, p_col]
                        # 确定电站破坏时间

                        damage_duration = frame_duration
                        unit_damage_loss_power_station = initial_unit_damage_loss_power_station * damage_duration
                        probability = calculate_power_station_damage_probability(depth)


                        damage_duration = frame_duration * (
                                    frame_count - power_station_damage_durations[ps_idx] + 1)
                        unit_damage_loss_power_station2 = initial_unit_damage_loss_power_station * damage_duration

                        distance = power_station.geometry.distance(road.geometry)

                        # 判断距离是否在影响范围内
                        if distance <= 50000:
                            # 计算影响值
                            influence = (facility_weights['power_station'] + facility_weights['road']) / (distance + 1)
                            # print(f"road:",influence)
                            # 如果影响值超过阈值，标记道路为受影响
                            if influence > 0.005:
                                if idx not in road_influence_durations:
                                    road_influence_durations[idx] = frame_count
                                # 考虑延时
                                if (frame_count - road_influence_durations[
                                    idx]) >= interval_road / interval:
                                    road_gdf.at[idx, 'status'] = 'Influenced'
                                    # color = 'black'


                                    if ps_idx in power_station_counts:
                                        power_station_counts[ps_idx] += 1
                                    else:
                                        power_station_counts[ps_idx] = 1
                                    # 连接边风险 乘以重要度
                                    current_connection_p_r_risk = (probability * unit_damage_loss_power_station2
                                                                   + sum_of_connection(
                                                probability * unit_damage_loss_power_station2,
                                                power_station_counts[ps_idx]))

                                    # 记录影响关系并存储详细信息
                                    influence_connections.append(([power_station.geometry.x, center_x],
                                                                  [power_station.geometry.y, center_y]))
                                    # 绘制影响关系的连接线
                                    ax.plot([power_station.geometry.x, center_x],
                                            [power_station.geometry.y, center_y],
                                            color='purple', linestyle='--', linewidth=1, zorder=3)

            road_previous_state[idx] = road_gdf.at[idx, 'status']

            # 根据道路状态设置颜色
            road_length = road.geometry.length  # 计算道路的长度
            if road_gdf.at[idx, 'status'] == 'Damaged':
                road_color = 'Maroon'
                road_damage_count += 1  # 增加损坏道路计数
                road_damage_length += road_length  # 累加损坏道路的长度
            elif road_gdf.at[idx, 'status'] == 'Influenced':
                road_color = 'black'
                road_influenced_count += 1  # 增加受影响道路计数
                road_influenced_length += road_length  # 累加受影响道路的长度

                max_depth = is_road_flooded(road, transform, water_depth)
                if idx not in road_influence_durations:
                    road_influence_durations[idx] = frame_count


                # 计算破坏持续时间并更新损失
                damage_duration = frame_duration
                road_length = road.geometry.length
                unit_damage_loss_road = initial_unit_damage_loss_road * (
                        damage_duration + road_length / 16.67)  # 该时刻损失
                total_road_influenced_loss += unit_damage_loss_road

                # color = 'Maroon'
                probability = calculate_road_damage_probability(max_depth)  # 基于最大水深计算破坏概率
                # 在道路中间绘制概率
                mid_x, mid_y = road.geometry.centroid.x, road.geometry.centroid.y

                # 计算风险
                risk = probability * unit_damage_loss_road

                total_road_risk += risk  # 加入当前帧风险

                damage_duration = frame_duration * (frame_count - road_influence_durations[idx] + 1)
                road_length = road.geometry.length
                unit_damage_loss_road2 = initial_unit_damage_loss_road * (
                        damage_duration + road_length / 16.67)  # 该时刻损失
                risk = probability * unit_damage_loss_road2
                road_gdf.at[idx, 'risk'] = risk
            else:
                road_color = 'moccasin'
                road_normal_count += 1  # 增加正常道路计数
                road_normal_length += road_length  # 累加正常道路的长度
            #print(f"Damaged roads: {road_damage_count}, Total Length: {road_damage_length:.2f} meters")
            #print(f"Influenced roads: {road_influenced_count}, Total Length: {road_influenced_length:.2f} meters")
            #print(f"Normal roads: {road_normal_count}, Total Length: {road_normal_length:.2f} meters")

            # 绘制道路
            if isinstance(road.geometry, LineString):
                x, y = zip(*road.geometry.coords)

            elif isinstance(road.geometry, MultiLineString):
                for line in road.geometry.geoms:
                    x, y = zip(*line.coords)
            ax.plot(x, y, color=road_color, linewidth=1)

        # 处理基站状态影响（从被破坏的电站或基站出发）
        for idx, base_station in base_station_gdf.iterrows():
            x, y = base_station.geometry.x, base_station.geometry.y
            row, col = coord_to_index(x, y, transform)

            # 保持上一帧的状态
            previous_status = base_station_previous_state.get(idx, 'Normal')

            # 如果上一帧已受损则保留该状态
            if previous_status in ['Damaged', 'Influenced']:
                base_station_gdf.at[idx, 'status'] = previous_status
                # 更新标注
                # if (x, y) in marked_base_station_texts:
                # marked_base_station_texts[(x, y)].set_text(f'Risk: {b_b_risk_values.get((x, y), 0):.2f}')

            if base_station_gdf.at[idx, "status"] == 'Normal':

                influenced_indices = affected_base_station.index  # 获取受影响建筑的索引
                if idx in influenced_indices:
                    base_station_gdf.at[idx, "status"] = "Damaged"  # 更新受影响建筑的状态
                if base_station_gdf.at[idx, "status"] == "Damaged":
                    if idx not in base_station_damage_durations:
                        base_station_damage_durations[idx] = frame_count
                    damage_duration = frame_duration * (frame_count - base_station_damage_durations[idx] + 1)
                    unit_damage_loss_base_station2 = initial_unit_damage_loss_base_station * damage_duration

                    probability = calculate_base_station_damage_probability(damage_duration)

                    risk = probability * unit_damage_loss_base_station2
                    base_station_gdf.at[idx, 'risk'] = risk

                if 0 <= row < grid_size[0] and 0 <= col < grid_size[1]:
                    depth = water_depth[row, col]
                    if depth > 0:
                        base_station_gdf.at[idx, 'status'] = 'Damaged'
                        # 记录基站破坏时间
                        if idx not in base_station_damage_durations:
                            base_station_damage_durations[idx] = frame_count

            if base_station_gdf.at[idx, "status"] == 'Normal':
                # 通过受损基站影响
                for other_idx, other_station in base_station_gdf.iterrows():
                    if base_station_gdf.at[other_idx,'status'] == 'Damaged' and other_idx != idx:
                        other_x, other_y = other_station.geometry.x, other_station.geometry.y
                        # other_row, other_col = coord_to_index(other_x, other_y, transform)
                        # other_depth = water_depth[other_row, other_col]
                        if other_idx not in base_station_damage_durations:
                            base_station_damage_durations[other_idx] = frame_count

                        damage_duration = (frame_count - base_station_damage_durations[
                                other_idx] + 1) * frame_duration
                        unit_damage_loss_base_station = initial_unit_damage_loss_base_station * damage_duration
                        probability = calculate_base_station_damage_probability(damage_duration)
                        distance = base_station.geometry.distance(other_station.geometry)
                        if distance <= 50000:
                            influence = facility_weights['base_station'] * 2 / (distance + 1)
                            if influence > 0.0035:
                                if idx not in base_station_damage_durations:
                                    base_station_damage_durations[idx] = frame_count
                                if (frame_count - base_station_damage_durations[
                                    idx]) >= interval_base_station / frame_duration:
                                    if other_idx in base_station_counts:
                                        base_station_counts[other_idx] += 1
                                    else:
                                        base_station_counts[other_idx] = 1
                                    # 计算链条风险

                                    damage_duration = frame_duration * (
                                                frame_count - base_station_damage_durations[other_idx] + 1)
                                    unit_damage_loss_base_station2 = initial_unit_damage_loss_base_station * damage_duration
                                    risk = probability * unit_damage_loss_base_station2
                                    base_station_gdf.at[idx, 'risk'] = risk
                                    # 连接边风险 乘以重要度
                                    current_connection_b_b_risk = (
                                            probability * unit_damage_loss_base_station2
                                            + sum_of_connection(
                                        0.7*probability * unit_damage_loss_base_station2,
                                        base_station_counts[other_idx]))
                                    # 记录新影响关系的直线并绘制
                                    influence_connections.append(
                                        ([other_station.geometry.x, base_station.geometry.x],
                                         [other_station.geometry.y, base_station.geometry.y]))
                                    ax.plot([other_station.geometry.x, base_station.geometry.x],
                                            [other_station.geometry.y, base_station.geometry.y],
                                            color='purple',
                                            linestyle='--',
                                            linewidth=1, zorder=3)
                                    # 更新基站状态
                                    base_station_gdf.at[idx, 'status'] = 'Influenced'
                                 

            # 更新上一帧状态
            base_station_previous_state[idx] = base_station_gdf.at[idx, 'status']

        # 在每帧的末尾绘制新生成的影响线条
        for new_connection in influence_connections:
            ax.plot(new_connection[0], new_connection[1], color='purple', linestyle='--', linewidth=1)

        base_station_damage_count = 0
        base_station_influenced_count = 0
        base_station_normal_count = 0
        # 更新并绘制基站状态（区分淹没破坏和受影响破坏）
        for idx, station in base_station_gdf.iterrows():
            x, y = station.geometry.x, station.geometry.y
            if base_station_gdf.at[idx,'status'] == 'Damaged':
                color = 'red'  # 淹没破坏为红色
                base_station_damage_count += 1  # 计数
                damage_duration = frame_duration
                unit_damage_loss_base_station = initial_unit_damage_loss_base_station * damage_duration
                total_base_station_damaged_loss += unit_damage_loss_base_station
                damage_duration = frame_duration * (frame_count - base_station_damage_durations[idx]+1)
                unit_damage_loss_base_station2 = initial_unit_damage_loss_base_station * damage_duration
                probability = calculate_base_station_damage_probability(damage_duration)
                # plt.text(x, y, f'{probability:.2f}', fontsize=3, color='black')
                risk = probability * unit_damage_loss_base_station
                total_base_station_risk += risk
                risk = probability * unit_damage_loss_base_station2
                base_station_gdf.at[idx, 'risk'] = risk

                # 记录标注灾害链风险到引发灾害的点
                if (x, y) in marked_base_station_texts:
                    current_connection_b_b_risk = (
                            probability * unit_damage_loss_base_station2
                            + sum_of_connection(
                        probability * unit_damage_loss_base_station2*0.7,
                        base_station_counts[idx]))

                    b_b_risk_values[(x, y)] = current_connection_b_b_risk
                    print(x,y,f'后',current_connection_b_b_risk)

            elif base_station_gdf.at[idx,'status'] == 'Influenced':
                color = 'orange'  # 受影响破坏为橙色
                base_station_influenced_count += 1  # 计数
                damage_duration = frame_duration
                unit_damage_loss_base_station = initial_unit_damage_loss_base_station * damage_duration
                total_base_station_influenced_loss += unit_damage_loss_base_station

                damage_duration = frame_duration * (
                            frame_count - base_station_damage_durations[idx] + 1)
                probability = calculate_base_station_damage_probability(damage_duration)
                unit_damage_loss_base_station2 = initial_unit_damage_loss_base_station * damage_duration
                risk = probability * unit_damage_loss_base_station2
                base_station_gdf.at[idx, 'risk'] = risk
                total_base_station_risk += risk
            else:
                color = 'Green'  # 正常状态
                base_station_normal_count += 1  # 计数
            x, y = station.geometry.x, station.geometry.y
            ax.scatter(x, y, color=color, s=10, marker='o', zorder=5)
        #print(f"Damaged base stations: {base_station_damage_count}")
        #print(f"Influenced base stations: {base_station_influenced_count}")
        #print(f"Normal base stations: {base_station_normal_count}")

        total_connection_risk = sum(b_b_risk_values.values()) + sum(p_b_risk_values.values()) + sum(
            p_r_risk_values.values())
        print(f'total_building_damaged_loss',total_building_damaged_loss)
        print(f'frame: {frame:03d}')
        print(f'frame_count: {frame_count}')


        record_frame_data(frame ,
                          #building_damage_area ,building_influenced_area ,building_normal_area,
                          #power_station_damage_count ,power_station_influenced_count ,power_station_normal_count ,
                          #base_station_damage_count ,base_station_influenced_count , base_station_normal_count,
                          road_damage_length,road_influenced_length,road_normal_length
                         ,road_damage_count,road_influenced_count,road_normal_count
                          )
 

        # 绘制图例
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Functional communication base station'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Damaged communication base station'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Affected communication base station'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', label='Functional power station'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', label='Damaged power station'),
            Patch(color='LightGrey', label='Functional area'),
            Patch(color='red', label='Inundated area'),
            Patch(color='gold', label='Power outage area'),
            Line2D([0], [0], color='moccasin', lw=2, label='Functional road'),
            Line2D([0], [0], color='Maroon', lw=2, label='Damaged road'),
            Line2D([0], [0], color='purple', lw=2, linestyle='--', label='Propagation chain'),
        ]
        # ax.legend(handles=legend_elements, loc='upper right')
        legend = ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 0), prop={'family': 'Times New Roman'})
        # 设置图例框的底色为白色
        legend.get_frame().set_facecolor('white')

        # 获取图例框的坐标和大小
        legend_bbox = legend.get_window_extent(ax.figure.canvas.get_renderer())

        # 转换图例坐标到轴的比例坐标（适用于 `ax.text`）
        legend_bbox = legend_bbox.transformed(ax.transAxes.inverted())

        # 确定文本位置，紧贴在图例框下方并右对齐
        text_x = legend_bbox.x1+0.0148  # 右对齐图例框
        text_y = legend_bbox.y1+0.059  # 紧贴图例框下方，距离控制在0.05以内

        # 保存当前帧为图片
        # 保存每一帧图片
        plt.savefig(os.path.join(output_folder, f"flood_frame_{frame:03d}.png"), dpi=600)

        # 保存基站状态
        base_station_filename = f"output_shp/base_station_status_frame_{frame:03d}.shp"
        base_station_gdf.to_file(base_station_filename, driver="ESRI Shapefile", encoding="utf-8")

        # 保存电站状态
        power_station_filename = f"output_shp/power_station_status_frame_{frame:03d}.shp"
        power_station_gdf.to_file(power_station_filename, driver="ESRI Shapefile", encoding="utf-8")
        # 保存建筑状态
        building_filename = f"output_shp/building_status_frame_{frame:03d}.shp"
        building_gdf.to_file(building_filename, driver="ESRI Shapefile", encoding="utf-8")
        # 保存道路状态
        road_filename = f"output_shp/road_status_frame_{frame:03d}.shp"
        road_gdf.to_file(road_filename, driver="ESRI Shapefile", encoding="utf-8")

        # 保存影响线条（不包含风险）
        save_influence_lines(frame, influence_connections)

        # 保存标注点为SHP文件
        # 创建空的列表用于保存标注点数据
        label_points = []
        facility_types = []  # 新增：用于保存设施类型（power_station 或 base_station）
        label_risks = {}
        x_coords = []
        y_coords = []

        # 处理 power_station 标注点
        for (ps_x, ps_y), text in marked_power_station_texts.items():
            # 获取当前的风险值
            risk_value_ps = float(text.get_text().split(":")[1].strip())

            # 创建一个点对象
            point_ps = Point(ps_x, ps_y)

            # 将点、设施类型和风险值存入列表
            label_points.append(point_ps)
            facility_types.append('power_station')
            label_risks[(ps_x, ps_y)] = risk_value_ps
            x_coords.append(point_ps.x)
            y_coords.append(point_ps.y)

        # 处理 base_station 标注点
        for (bs_x, bs_y), text in marked_base_station_texts.items():
            # 获取当前的风险值
            risk_value_bs = float(text.get_text().split(":")[1].strip())

            # 创建一个点对象
            point_bs = Point(bs_x, bs_y)

            # 将点、设施类型和风险值存入列表
            label_points.append(point_bs)
            facility_types.append('base_station')
            label_risks[(bs_x, bs_y)] = risk_value_bs
            x_coords.append(point_bs.x)
            y_coords.append(point_bs.y)
        if label_points:
            # 创建GeoDataFrame并添加坐标、设施类型和风险值
            label_gdf = gpd.GeoDataFrame({
                'geometry': label_points,
                'facility_type': facility_types,  # 新增：设施类型列
                'risk_value': [label_risks[point.coords[0]] for point in label_points],
                'X': x_coords,
                'Y': y_coords
            }, crs="EPSG:32650")  # 假设使用WGS 84坐标系
            # 设置输出文件路径
            label_points_filename = f"output_shp/label_points_frame_{frame:03d}.shp"

            # 导出为Shapefile
            label_gdf.to_file(label_points_filename, driver="ESRI Shapefile", encoding="utf-8")
        else:

            print(f"No label_points to save for frame {frame}.")

        # 获取结束时间
        end = time.perf_counter()
        # 计算运行时间
        runTime = end - start
        # 输出运行时间
        print("运行时间：", runTime, "秒")

    # 获取水深数据文件列表
    water_depth_files = sorted(os.listdir(water_depth_folder))
    num_frames = len(water_depth_files)  # 动态获取帧数

    # 生成动画
    animation = FuncAnimation(fig, update_flood, frames=range(num_frames), repeat=False)

    # animation = FuncAnimation(fig, update_flood, frames=range(0, 10), repeat=False)
    animation.save(os.path.join(output_folder, 'flood_simulation.gif'), writer='pillow', fps=2)

    plt.close(fig)
    return "Simulation completed successfully."

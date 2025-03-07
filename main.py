from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import zipfile
import geopandas as gpd
import rasterio
from flood_simulation_code import flood_simulation
import time
from rasterio.warp import calculate_default_transform, reproject
from pyproj import CRS
import shutil


app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_files'
INFRASTRUCTURE_FOLDER = os.path.join(UPLOAD_FOLDER, 'infrastructure')
FLOOD_DEPTHS_FOLDER = os.path.join(UPLOAD_FOLDER, 'flood_depths')
OUTPUT_FOLDER = 'simulation_output'
OUTPUT_SHP_FOLDER = 'output_shp'
FLOOD_DEPTHS_FOLDER_NEW = 'new_flood_depths'

# 自动创建文件夹
os.makedirs(INFRASTRUCTURE_FOLDER, exist_ok=True)
os.makedirs(FLOOD_DEPTHS_FOLDER, exist_ok=True)
os.makedirs(FLOOD_DEPTHS_FOLDER_NEW, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_SHP_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('upload.html')  # 渲染上传页面

def clear_directory(directory):
    """ 清空目录中的所有文件和子文件夹 """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


@app.route('/upload', methods=['POST'])
def upload_files():
    # 获取开始时间
    start = time.perf_counter()

    # 清空旧数据
    clear_directory(INFRASTRUCTURE_FOLDER)
    clear_directory(FLOOD_DEPTHS_FOLDER)
    clear_directory(FLOOD_DEPTHS_FOLDER_NEW)

    # 获取上传的ZIP文件
    infrastructure_zip = request.files['infrastructure_zip']
    flood_depths_zip = request.files['flood_depths_zip']

    # 保存基础设施ZIP文件
    infra_zip_path = os.path.join(UPLOAD_FOLDER, infrastructure_zip.filename)
    infrastructure_zip.save(infra_zip_path)

    # 保存水深ZIP文件
    flood_depths_zip_path = os.path.join(UPLOAD_FOLDER, flood_depths_zip.filename)
    flood_depths_zip.save(flood_depths_zip_path)

    # 解压基础设施ZIP文件到专用文件夹
    with zipfile.ZipFile(infra_zip_path, 'r') as zip_ref:
        zip_ref.extractall(INFRASTRUCTURE_FOLDER)

    # 解压水深ZIP文件到专用文件夹
    with zipfile.ZipFile(flood_depths_zip_path, 'r') as zip_ref:
        zip_ref.extractall(FLOOD_DEPTHS_FOLDER)


    # 处理基础设施数据
    buildings_gdf = gpd.read_file(os.path.join(INFRASTRUCTURE_FOLDER, 'infrastructure/buildings/buildings.shp'))
    roads_gdf = gpd.read_file(os.path.join(INFRASTRUCTURE_FOLDER, 'infrastructure/roads/roads.shp'))
    power_stations_gdf = gpd.read_file(os.path.join(INFRASTRUCTURE_FOLDER, 'infrastructure/power_stations/power_stations.shp'))
    base_stations_gdf = gpd.read_file(os.path.join(INFRASTRUCTURE_FOLDER, 'infrastructure/base_stations/base_stations.shp'))
    ##修改水深投影
    # 输入文件夹路径，包含多个水深数据文件（.tif）
    input_folder = FLOOD_DEPTHS_FOLDER
    output_folder = FLOOD_DEPTHS_FOLDER_NEW

    # 目标坐标系 EPSG:32650 (WGS 1984 UTM 50N)
    target_crs = CRS.from_epsg(32650)

    # 遍历输入文件夹中的所有 TIF 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):  # 确保是 TIF 文件
            water_depth_file = os.path.join(input_folder, filename)

            # 读取栅格数据文件
            with rasterio.open(water_depth_file) as src:
                # 获取原始坐标系
                src_crs = src.crs

                # 计算转换后的栅格大小和仿射变换矩阵
                transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height,
                                                                       *src.bounds)

                # 更新元数据
                metadata = src.meta.copy()
                metadata.update({
                    'crs': target_crs.to_string(),  # 设置为目标坐标系
                    'transform': transform,  # 更新变换矩阵
                    'width': width,  # 更新宽度
                    'height': height  # 更新高度
                })

                # 在内存中进行重投影并保存到输出文件夹
                output_file = os.path.join(output_folder, filename)  # 输出文件路径
                with rasterio.open(output_file, 'w', **metadata) as dst:
                    for i in range(1, src.count + 1):  # 遍历每个波段
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src_crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=rasterio.enums.Resampling.nearest)  # 使用最近邻插值

    # 运行灾害链动态模拟
    result = flood_simulation(buildings_gdf, roads_gdf, base_stations_gdf, power_stations_gdf,
                              OUTPUT_FOLDER,OUTPUT_SHP_FOLDER, FLOOD_DEPTHS_FOLDER_NEW)
    # 获取结束时间
    end = time.perf_counter()
    # 计算运行时间
    runTime2 = end - start
    # 输出运行时间
    print("运行时间：", runTime2, "秒")

    return jsonify({"message": "Simulation complete", "result": result}), 200


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

import os
import yaml

from PIL import Image
from exif import Image as ExifImage
from math import tan, atan, radians
from shapely.geometry import Polygon
from shapely.validation import make_valid
from geopy.distance import geodesic


def validate_inputs(lat, lon, gca_width, gca_height):
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} is out of bounds.")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude {lon} is out of bounds.")
    if gca_width <= 0 or gca_height <= 0:
        raise ValueError("Ground coverage width and height must be positive.")

# 验证和修复多边形
def validate_polygon(polygon):
    if not polygon.is_valid:
        print(f"Invalid Polygon detected. Attempting to fix...")
        polygon = make_valid(polygon)  # 修复多边形
    return polygon

def dms_to_decimal(dms, ref):
    """将DMS格式（度、分、秒）转换为十进制格式"""
    degrees = dms[0] + dms[1] / 60 + dms[2] / 3600
    if ref in ['S', 'W']:  # 南纬或西经需要是负数
        degrees = -degrees
    return degrees

def compute_fov(sensor_width, sensor_height, focal_length):
    """计算相机的水平和垂直视场角（FOV）"""
    fov_horizontal = 2 * atan(sensor_width / (2 * focal_length))
    fov_vertical = 2 * atan(sensor_height / (2 * focal_length))
    return fov_horizontal, fov_vertical

def compute_ground_coverage(fov_horizontal, fov_vertical, altitude):
    """计算图像的地面覆盖宽度和高度"""
    width = 2 * altitude * tan(fov_horizontal / 2)
    height = 2 * altitude * tan(fov_vertical / 2)
    return width, height

def get_image_metadata(image_path):
    """读取图像的EXIF元数据，并转换GPS数据"""
    with open(image_path, 'rb') as img_file:
        img = ExifImage(img_file)
        if not img.has_exif:
            raise ValueError(f"Image {image_path} does not have EXIF metadata.")
        
        # 获取GPS数据并转换
        lat = dms_to_decimal(img.gps_latitude, img.gps_latitude_ref)
        lon = dms_to_decimal(img.gps_longitude, img.gps_longitude_ref)
        altitude = float(img.gps_altitude)
        focal_length = float(img.focal_length)
        
        return {
            "latitude": lat,
            "longitude": lon,
            "altitude": altitude,
            "focal_length": focal_length,
            "width": img.pixel_x_dimension,
            "height": img.pixel_y_dimension
        }

def create_polygon(lat, lon, gca_width, gca_height):
    """基于经纬度中心点和地面覆盖面积创建多边形"""
    validate_inputs(lat, lon, gca_width, gca_height)
    # 四个角点计算
    top_left = geodesic(kilometers=gca_height / 2).destination((lat, lon), bearing=0)
    bottom_left = geodesic(kilometers=gca_height / 2).destination((lat, lon), bearing=180)
    top_right = geodesic(kilometers=gca_width / 2).destination((lat, lon), bearing=90)
    bottom_right = geodesic(kilometers=gca_width / 2).destination((lat, lon), bearing=270)
    polygon = Polygon([
        (top_left.latitude, top_left.longitude),
        (top_right.latitude, top_right.longitude),
        (bottom_right.latitude, bottom_right.longitude),
        (bottom_left.latitude, bottom_left.longitude),
    ])
    return validate_polygon(polygon)  # 验证和修复多边形

def calculate_overlap(poly1, poly2):
    """计算两个多边形的重叠度"""
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return intersection_area / union_area if union_area > 0 else 0

def sample_images(image_folder, overlap_threshold=0.1, sensor_width=22, sensor_height=22):
    """主函数：对图像进行采样"""
    sampled_images = []  # 采样后的图像路径
    sampled_polygons = []  # 采样后的多边形列表
    
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')])
    if not image_files:
        raise ValueError("No JPG images found in the specified folder.")
    
    # 逐一读取并处理图像
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        metadata = get_image_metadata(image_path)
        
        # 计算FOV和地面覆盖范围
        fov_horizontal, fov_vertical = compute_fov(
            sensor_width, # 假设标准35mm传感器宽度
            sensor_height, # 假设标准35mm传感器高度
            focal_length=metadata["focal_length"]
        )
        gca_width, gca_height = compute_ground_coverage(
            fov_horizontal, fov_vertical, metadata["altitude"]
        )
        
        # 创建当前图像的地面覆盖多边形
        try:
            current_polygon = create_polygon(
                metadata["latitude"], metadata["longitude"], gca_width, gca_height
            )
        except Exception as e:
            print(f"Skipping image {image_file} due to polygon error: {e}")
            continue
        
        # 如果是第一个图像，直接加入采样集合
        if idx == 0:
            sampled_images.append(image_path)
            sampled_polygons.append(current_polygon)
            continue
        
        # 判断与已有采样图像的重叠度
        is_overlapping = any(
            calculate_overlap(current_polygon, sampled_polygon) > overlap_threshold
            for sampled_polygon in sampled_polygons
        )
        
        # 如果不重叠，加入采样集合
        if not is_overlapping:
            sampled_images.append(image_path)
            sampled_polygons.append(current_polygon)
    
    return sampled_images

# 调用函数并保存采样结果
if __name__ == "__main__":
    with open('config/config.yml', 'r', encoding="utf-8") as file:
            user_config = yaml.safe_load(file)
    sampled_images = sample_images(image_folder=user_config["image_folder"], 
                                   
                                   overlap_threshold=user_config["gps_threshold"], 
                                   
                                   sensor_width=user_config["sensor_width"], 
                                   
                                   sensor_height=user_config["sensor_height"])

    # 输出采样结果
    print("Sampled Images:")
    for img in sampled_images:
        print(img)

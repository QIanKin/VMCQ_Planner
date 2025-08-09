import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # 替换为您的包名
    pkg_name = 'vmc_minco_nav' 

    # Rviz配置文件的路径 (您可以基于旧的rviz配置新建一个)
    rviz_config_file = os.path.join(
        get_package_share_directory(pkg_name),
        'rviz', 'verification.rviz') 

    # 随机圆柱体障碍物生成器节点
    obstacle_generator_node = Node(
        package=pkg_name,
        executable='random_cylinder_publisher',
        name='random_cylinder_publisher',
        output='screen',
        parameters=[{
            'map_size.x': 20.0,
            'map_size.y': 20.0,
            'obstacles.number': 20,
            'obstacles.radius_min': 0.5,
            'obstacles.radius_max': 2.0,
            'frame_id': 'map'
        }]
    )
    
    # 模拟机器人起始位置的TF发布节点
    fake_robot_tf_node = Node(
        package=pkg_name,
        executable='fake_robot_tf_publisher',
        name='fake_robot_tf_publisher',
        output='screen',
        parameters=[{'start.x': 1.0, 'start.y': 1.0}] 
    )

    # 验证导航器节点
    verification_navigator_node = Node(
        package=pkg_name,
        executable='verification_navigator',
        name='verification_navigator',
        output='screen',
        # 在这里加载您的所有VMC和MINCO参数
        # parameters=[os.path.join(get_package_share_directory(pkg_name), 'config', 'params.yaml')]
    )

    # Rviz节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    return LaunchDescription([
        obstacle_generator_node,
        fake_robot_tf_node,
        verification_navigator_node,
        rviz_node
    ])
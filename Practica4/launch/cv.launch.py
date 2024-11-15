# Copyright (c) 2023 José Miguel Guerrero Hernández
#
# Licensed under the Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    darknet_dir = get_package_share_directory('darknet_ros')

    darknet_cmd = IncludeLaunchDescription(
      PythonLaunchDescriptionSource(os.path.join(darknet_dir,
                                    'launch', 'darknet_ros.launch.py')))

    
    cv_cmd = Node(
        package='practicafinal-grupo15',
        namespace='computer_vision',
        executable='cv_program',
        output='both',
        emulate_tty=True,
        # Set to True to process just if there is a subscription,
        # False to process always
        parameters=[
            {"check_subscription_count": False}
        ],
        # Use topics from robot
        remappings=[
            ('/camera_info', '/head_front_camera/rgb/camera_info'),
            ('/image_rgb_in', '/head_front_camera/rgb/image_raw'),
            ('/image_depth_in', '/head_front_camera/depth_registered/image_raw'),
            ('/pointcloud_in', '/head_front_camera/depth_registered/points'),
        ],
    )
    ld = LaunchDescription()
    # ld.add_action(darknet_cmd)
    ld.add_action(cv_cmd)
    return ld
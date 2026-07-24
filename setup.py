import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'm300_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lucasm',
    maintainer_email='marounlucas@gmail.com',
    description='Simulation Environment for M300 aircraft',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'traj_pub = scripts.trajectory.trajectory_node:main',
            'ctrl_sub = scripts.control.control_node:main',
            'dyn_pub = scripts.dynamic.dynamic_node:main',
            'joy_sub = scripts.manual_sim.joy_mapper:main',
            'gui_node = scripts.gui.main:main',
        ],
    },
)

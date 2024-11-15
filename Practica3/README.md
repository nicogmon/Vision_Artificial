[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Tg9RTWQ_)
# Computer Vision Examples

![distro](https://img.shields.io/badge/Ubuntu%2022-Jammy%20Jellyfish-green)
![distro](https://img.shields.io/badge/ROS2-Humble-blue)
[![humble](https://github.com/jmguerreroh/computer_vision/actions/workflows/master.yaml/badge.svg?branch=humble)](https://github.com/jmguerreroh/computer_vision/actions/workflows/master.yaml)

This project contains code examples created in Visual Studio Code for Computer Vision using C++ & OpenCV & Point Cloud Library (PCL) in ROS 2. These examples are created for the Computer Vision Subject of Robotics Software Engineering Degree at URJC.

This package is recommended to use with the [TIAGO](https://github.com/jmguerreroh/tiago_simulator) simulator.

# Installation 

You need to have previously installed ROS 2. Please follow this [guide](https://docs.ros.org/en/humble/Installation.html) if you don't have it.
```bash
source /opt/ros/humble/setup.bash
```

Clone the repository to your workspace:
```bash
mkdir -p ~/cv_ws/src
cd ~/cv_ws/src/
git clone https://github.com/jmguerreroh/tiago_simulator.git
cd ~/cv_ws/
rosdep install --from-paths src --ignore-src -r -y
```

# Building project

```bash
colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF
``` 
# Run

Execute:
```bash
ros2 launch computer_vision cv.launch.py
```
If you want to use your own robot, in the launcher, change the topic names to match the robot topics.

## FAQs:

* /usr/bin/ld shows libraries conflicts between two versions:

Probably you have installed and built your own OpenCV version, rename your local folder:
```bash
mv /usr/local/lib/cmake/opencv4 /usr/local/lib/cmake/oldopencv4
```
## Questions

1. En el apartado 2, se han generado una serie de puntos cuya distancia es con respecto a
basefootprint. ¿Con respecto a qué frame de la cámara has realizado los cálculos para
la obtención de los parámetros extrínsecos? Justifica tu respuesta.

Hemos realizado los cálculos respecto al frame Head_front_camera_rgb_optical_frame.
Nuestra elección se basa en la extracción de los parámetros intrínsecos ya que es este frame al que se refieren dichos parámetros.  

2. Una vez aplicado RANSAC en PCL. ¿Cómo has identificado que el plano obtenido
pertenece en realidad a una de las baldas de las estanterías?

Para la identificación de los planos correspondientes a las baldas he aislado los planos horizontales de el total de planos obtenidos.
Dado que con el filtro de color hemos aislado los componentes amarillos pertenencientes a la estrucutra general de las baldas solo nos tendría que preocupar distingir entre los planos verticales o diagonales que se pudieran generar en base a los tubos que conforman la estructura exterior de sujección de las baldas. 

## About

This is a project made by [José Miguel Guerrero], Associate Professor at [Universidad Rey Juan Carlos].

Copyright &copy; 2024.

[![Twitter](https://img.shields.io/badge/follow-@jm__guerrero-green.svg)](https://twitter.com/jm__guerrero)

## License

Shield: 

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

[Universidad Rey Juan Carlos]: https://www.urjc.es/
[José Miguel Guerrero]: https://sites.google.com/view/jmguerrero

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6bfcAzJo)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13786573&assignment_repo_type=AssignmentRepo)
# Computer Vision Examples

![distro](https://img.shields.io/badge/Ubuntu%2022-Jammy%20Jellyfish-green)
![distro](https://img.shields.io/badge/ROS2-Humble-blue)
[![humble](https://github.com/jmguerreroh/computer_vision/actions/workflows/master.yaml/badge.svg?branch=humble)](https://github.com/jmguerreroh/computer_vision/actions/workflows/master.yaml)

This project contains code examples created in Visual Studio Code for Computer Vision using C++ & OpenCV & Point Cloud Library (PCL) in ROS2. These examples are created for the Computer Vision Subject of Robotics Software Engineering Degree at URJC.

This package is recommended to use with the [TIAGO](https://github.com/jmguerreroh/tiago_simulator) simulator.

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
**Question 1**:
In the representation of hsv and hsi the values of bgr are used, equivalent to h=h=b, s=s=g and v=i=r. Therefore, given that the value of h is calculated in the same way, they disappear in the subtraction, thus not obtaining a blue colour in the image shown. In the case of saturation(s), it should be calculated in the same way but for internal reasons of opencv, the calculation is not the same as the manual one. This is why we mainly get an image with green tones.  And finally, the v and i which are also not calculated the same, we get red tones.

Furthermore, the image is almost completely black because the predominant colour is white, so the subtraction of Hue (dominant colour) is cancelled out.

![image](https://github.com/computer-vision-urjc/practica1-grupo15/assets/102520722/8368ef58-16cd-4a53-a145-0842f56112d5)

**Question 2**:
In option 4 we are keeping the horizontal frequencies, which generates a smoothing effect in which more colour distortion is created between white and black. By passing a threshold of 0.6 to this smoothed image, most of the pixels exceed the threshold value generating a mostly black image. 

In contrast, in option 5, the horizontals are removed, leaving the edges black and the rest a uniform shade of grey. By applying the threshold, it could be said that only the edges that were black remain black and the rest are white.

When we apply the logical operator or between the two images we obtain an image very similar to the thresholded image of option 5 with small differences in the edges. These differences are due to the fact that in the thresholded image of option 4, there is an area with a white pixel that in the thresholded option 5 is black.


**!! COMMITS ARE IN THE BRANCH 'Desarrollo' !!**

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

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/ixoJ3y_C)
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


## Questions
**Question 1**:
. Adjunta una captura de los histogramas obtenidos en la opción 1 cuando los valores
mínimo y máximo de la contracción son 125 y 126 respectivamente, y explica
brevemente el comportamiento de cada histograma en dicha imagen.  
![image](https://github.com/computer-vision-urjc/practica2-grupo15/assets/102520722/f81741fc-9656-4db2-9479-2b0396888e16)

Habiendo sido tomada la captura de los histogramas con el robot tiago en la posición utilizada para el ejercicio 4, con una iluminación adecuada que permita la vilisibilidad de todos los colores de la pared, del ventanal y de la mesa con el ordenador, procedemos a explicar el resultado de los histogramas.

En primer lugar, vemos el histograma original de la imagen en escala de grises en el observamos un pico inicial debido a la predominancia del azul en diferentes tonalidades. Dado que en el cálculo de opencv para obtener la imagen en escala de grises, el azul es el color que menos peso tiene y por tanto aparece en el histograma en la parte izquierda ya que su valor es inferior al de los colores rojo y verde.

Por la parte contraria, observamos también valores muy altos en esa parte derecha. Considero que esto es debido a que una gran parte de la imagen la observamos de color marrón, la cual estará formada por estos dos colores rojo y verde. Además, contamos con las líneas verdes de la pared y dado que estos dos colores tienen un mayor peso, en la conversión se verán reflejados en esta parte derecha del histograma.

En segundo lugar, observamos el histograma correspondiente a la contracción entre los valores deseados. En este caso particular que se nos pide, veremos que hemos contraído el histograma entre 125 y 126 y por ello, observamos como todos los valores se concentran en ese pequeño intervalo situado en el centro del histograma. Esto es debido a que el cálculo realizado, lo que conseguirá es convertir los valores de cada píxel de la imagen para que de manera proporcional queden situados dentro del margen indicado.

De los dos siguientes histogrmas solo conseguimos apreciar el amarillo ya que el cyan correspondiente a la expansión de la resta del punto 4 no parece surtir efecto. Esto es suponiendo que en mi escenario la resta ya continene valores en los máximos y mínimos utilizados en la expansión y, por tanto, ambos histogramas coinciden.

Sobre el histograma en si correspondiente a la resta, observamos la mayoría de valores en 0 debido a al sentido de la resta, ya que el primer término rara vez será mayor que cero. Esto es debido a la contracción realizada que solo deja valores mayores que 0 en en el intervalo de una unidad descrito. 

Por último, en el histograma de color verde observamos la ecualización del anterior y cómo afecta principalmente en los valores bajos que se comentaban antes. Se puede ver como intenta distribuirlos de manera uniforme en esos picos que se muestran en la base de la imagen. En la izquierda encontramos un gran pico que también aparece en los dos anteriores histogramas y que no consigue repartir dado a su gran peso.  


**Question 2**
¿Es posible acotar la dirección de las líneas detectadas en la transformada de Hough?   
En caso afirmativo, ¿cómo? Justifique la/s respuesta/s.  
La propia función de Hough no nos da ninguna herramienta para poder acotar dichas líneas pero en cambio sí podemos usar los parámetros de salida de dichas líneas.  
La transformada de Hough tiene dos parámetros principales: la distancia desde el origen al punto más cercano de la línea (r), y el ángulo (θ) que forma la línea r con el eje x. Al ajustar el rango de valores posibles para θ, se puede limitar la dirección de las líneas que serán detectadas.  
![Screenshot from 2024-03-29 13-24-57](https://github.com/computer-vision-urjc/practica2-grupo15/assets/102520722/2e7582a0-5ed2-4654-bb58-1cd3815d0b94)  
Por tanto si quisiéramos por ejemplo detectar solo las líneas horizontales podríamos utilizar un código parecido al siguiente:    

```c++
vector<Vec2f> lines;
HoughLines(edges, lines, 1, CV_PI / 180, 100);

// Definir el rango de ángulos para líneas horizontales (por ejemplo, +-10 grados respecto a la horizontal)
float angle_range = CV_PI / 18;  // 10 grados en radianes
float theta_min = CV_PI / 2 - angle_range;
float theta_max = CV_PI / 2 + angle_range;

// Filtrar las líneas detectadas para mantener solo las horizontales
vector<Vec2f> filtered_lines;
for (size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0];
    float theta = lines[i][1];
    if (theta >= theta_min && theta <= theta_max) {
        filtered_lines.push_back(lines[i]);
    }
}
```

Utilizamos un ángulo de 90 grados(pi/2) ya que consideramos que la línea será horizontal si esta es paralela al eje x y por tanto el ángulo formado por r con el eje x será de 90º.

## FAQs:

* /usr/bin/ld shows libraries conflicts between two versions:

Probably you have installed and built your own OpenCV version, rename your local folder:
```bash
mv /usr/local/lib/cmake/opencv4 /usr/local/lib/cmake/oldopencv4
```

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

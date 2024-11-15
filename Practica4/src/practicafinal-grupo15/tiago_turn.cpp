#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <unistd.h>
#include <termios.h>
#include <iostream>
#include "geometry_msgs/msg/twist.hpp"

class ArrowKeyPublisher : public rclcpp::Node {
public:
    ArrowKeyPublisher()
    : Node("arrow_key_publisher") {
        arrow_key_pub_ = this->create_publisher<std_msgs::msg::String>("arrow_keys", 10);
        vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("key_vel", 10);
        
    }

    void run() {
        struct termios oldt, newt;
        char ch;
        int modo = 0;
        if (first == 0){
            std::cout << "Pulse 3 para manejar el robot con las flechas del teclado y las acciones previstas para esta opcion\n"<< std::endl;
            std::cout << "Pulse 4 para manejar el robot como con key_teleop y pulse l una vez para descubrir su localizacion" << std::endl;
            first = 1;
        }

        // Guardar la configuración actual de la terminal
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;

        // Deshabilitar la impresión de caracteres en la terminal
        newt.c_lflag &= ~(ICANON | ECHO);

        // Aplicar la nueva configuración
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        
        while (rclcpp::ok()) {
            ch = getchar();
            if (ch == '3'){
                std::cout << "Modo opcion 3" << std::endl;
                modo = 0;
            }
            if (ch == '4'){
                std::cout << "Modo opcion 4" << std::endl;
                modo = 1;
            }
            if(modo == 0){
                if (ch == 27 && getchar() == '[') { // Si es una secuencia de escape
                    char arrow = getchar(); // Obtener el código de flecha
                    if (arrow == 'A') { // Flecha arriba
                        publishArrowKey("Up");
                        
                        auto msg = std::make_unique<geometry_msgs::msg::Twist>();
                        msg->linear.x = 0.8;
                        vel_pub_->publish(std::move(msg));
                    } else if (arrow == 'B') { // Flecha abajo
                        publishArrowKey("Down");
                        
                        auto msg = std::make_unique<geometry_msgs::msg::Twist>();
                        msg->linear.x = -0.8;
                        vel_pub_->publish(std::move(msg));
                    } else if (arrow == 'C') { // Flecha derecha
                        publishArrowKey("Right");
                        publishForDuration(0);
                        std::cout << "Right" << std::endl;
                    } else if (arrow == 'D') { // Flecha izquierda
                        publishArrowKey("Left");
                        std::cout << "Left" << std::endl;
                        publishForDuration(1);
                    }
                }
            }
            else if (modo == 1){
                if (ch == 27 && getchar() == '[') { // Si es una secuencia de escape
                    char arrow = getchar(); // Obtener el código de flecha
                    if (arrow == 'A') { // Flecha arriba
                        publishArrowKey("Up");
                        
                        auto msg = std::make_unique<geometry_msgs::msg::Twist>();
                        msg->linear.x = 0.8;
                        vel_pub_->publish(std::move(msg));
                    } else if (arrow == 'B') { // Flecha abajo
                        publishArrowKey("Down");
                        
                        auto msg = std::make_unique<geometry_msgs::msg::Twist>();
                        msg->linear.x = -0.8;
                        vel_pub_->publish(std::move(msg));
                    } else if (arrow == 'C') { // Flecha derecha
                        auto msg = std::make_unique<geometry_msgs::msg::Twist>();
                        msg->angular.z = -1;
                        vel_pub_->publish(std::move(msg));

                    } else if (arrow == 'D') { // Flecha izquierda
                        auto msg = std::make_unique<geometry_msgs::msg::Twist>();
                        msg->angular.z = 1;
                        vel_pub_->publish(std::move(msg));
                    }
                }
                if (ch == 'l') {
                    publishArrowKey("l");
                }
            }
        }

        // Restaurar la configuración original de la terminal
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    }

private:
    int first = 0;
    void publishArrowKey(const std::string& key) {
        auto msg = std::make_unique<std_msgs::msg::String>();
        msg->data = key;
        arrow_key_pub_->publish(std::move(msg));
    }

     void publishForDuration(int direction) {
        auto start_time = std::chrono::steady_clock::now();
        while (rclcpp::ok()) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            if (elapsed_time >= 2) {
                break; // Termina la publicación después de 1.5 segundos
            }

            auto msg = std::make_unique<geometry_msgs::msg::Twist>();
            if (direction == 0) {
                msg->angular.z = -1;
            } else if (direction == 1) {
                msg->angular.z = 1;
            }
            
            vel_pub_->publish(std::move(msg));

            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Publica cada 0.1 segundos
        }
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr arrow_key_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr vel_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArrowKeyPublisher>();
    node->run();
    rclcpp::shutdown();
    return 0;
}


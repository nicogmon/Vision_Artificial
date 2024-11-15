/*
 Copyright (c) 2023 José Miguel Guerrero Hernández

 Licensed under the Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License;
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     https://creativecommons.org/licenses/by-sa/4.0/

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
/*  Grupo 0: Ana Martinéz y Nicolás García  
Partes implementadas: - Opciones 1, 2, 3 y 4
*/ 
#include "computer_vision/CVSubscriber.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

#include <unistd.h>
#include <termios.h>
#include <iostream>


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace dnn;


bool first = true;

string winname = "Practica_Final";
int option = 0;

// option3 variables
vector<Scalar> colors;
Mat old_frame, old_gray;
vector<Point2f> p0, p1;
Mat mask;
Mat mapa;
int UP = 0;
int DOWN = 1;
int LEFT = 2;
int RIGHT = 3;
int contUP = 0;
int contDOWN = 0;
int contLEFT = 0;
int contRIGHT = 0;
int direction = UP;
Point2d position = Point2d(300,300);
int flag_recalc = 0;



// <------- OPCION 4 -------->
// opcion 4 variables
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;
Net net;
static const string kWinName = "Deep learning object detection in OpenCV";
std::clock_t start_time;
std::vector<float> conc_perc;
bool turn_flag = false;
bool first_time = true;
struct termios oldt, newt;
char ch;

// <-------- estancias -------->
const std::vector<std::vector<std::string>> kitchen = {
  {"fridge", "sink", "cup", "spoon"},
  {"chair", "cup"},
};

const std::vector<cv::Point2d> kitchen_points = {
  cv::Point2d(8.4 , -1),
  cv::Point2d(8, -4.82),
  cv::Point2d(8.6, -1.95),
  cv::Point2d(8.8, -4.86),
  cv::Point2d(6.69, 0.96),
  cv::Point2d(2.38,-0.78)
};
const std::vector<std::vector<std::string>> living_room = {
  {"sofa", "cup"},
  {"sport ball", "chair"}
};
const std::vector<cv::Point2d> living_room_points = {
  cv::Point2d(0.33,-2.5),
  cv::Point2d(2.38,-0.78),
  cv::Point2d(3.32, 4.05),
  cv::Point2d(-1.22,4.14)
};

const std::vector<std::vector<std::string>> gym = {
  {"sports ball", "cup", "chair"},
  {"sofa"}
};
const std::vector<std::vector<std::string>> dinning_room = {
  {"dining table", "chair"},
  {"cup" }
};
const std::vector<std::vector<std::string>> bedroom = {
  {"bed", "cup", "chair", "person"},
  {"sofa"}
};
const std::vector<std::vector<std::string>> dresser = {
  {"sport ball", "chair", "toilet", "person"},
  {""}
};

using room = std::vector<std::vector<std::string>>;
std::vector<room> rooms = {kitchen, living_room, gym, dinning_room, bedroom, dresser};
std::vector<string> room_names = {"kitchen", "living_room", "gym", "dinning_room", "bedroom", "dresser"};
std::set<string> near_objects;
std::set<double> near_objects_distance;
std::set<string> far_objects;
std::set<double> far_objects_distance;


void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat & frame, Point2d center, Mat & out_image_depth)
{
  //Draw a rectangle displaying the bounding box
  rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

  //Get the label for the class name and its confidence
  string label = format("%.2f", conf);
  if (!classes.empty()) {
    CV_Assert(classId < (int)classes.size());
    label = classes[classId] + ":" + label + " " + to_string(out_image_depth.at<float>(center.y, center.x)) + "m";
    
  }

  //Display the label at the top of the bounding box
  int baseLine;
  Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = max(top, labelSize.height);
  rectangle(
    frame, Point(left, top - round(1.5 * labelSize.height)),
    Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
  putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

void trilateracion(cv::Point2d A, cv::Point2d B, cv::Point2d C, double dA, double dB, double dC){
  double x, y;

  // Calcular las diferencias de coordenadas
  double xDiffAB = B.x - A.x;
  double xDiffAC = C.x - A.x;
  double yDiffAB = B.y - A.y;
  double yDiffAC = C.y - A.y;

  // Calcular las ecuaciones intermedias
  double divAB = pow(dA, 2) - pow(dB, 2) + pow(B.x, 2) - pow(A.x, 2) + pow(B.y, 2) - pow(A.y, 2);
  double divAC = pow(dA, 2) - pow(dC, 2) + pow(C.x, 2) - pow(A.x, 2) + pow(C.y, 2) - pow(A.y, 2);

  // Calcular las coordenadas del punto objetivo
  x = (divAB * yDiffAC - divAC * yDiffAB) / (2 * (xDiffAB * yDiffAC - xDiffAC * yDiffAB));
  y = (divAB * xDiffAC - divAC * xDiffAB) / (2 * (yDiffAB * xDiffAC - yDiffAC * xDiffAB));

  std::cout << "Las coordenadas del punto objetivo son: (" << x << ", " << y << ")" << std::endl;

}

void postprocess(Mat & frame, const vector<Mat> & outs, Mat & out_image_depth)
{
  vector<int> classIds;
  vector<float> confidences;
  vector<Rect> boxes;
  vector<Point2d> centers;
  
  for (size_t i = 0; i < outs.size(); ++i) {
    // Scan through all the bounding boxes output from the network and keep only the
    // ones with high confidence scores. Assign the box's class label as the class
    // with the highest score for the box.
    
    float * data = (float *)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
      Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      Point classIdPoint;
      double confidence;
      // Get the value and location of the maximum score
      minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > confThreshold) {
        int centerX = (int)(data[0] * frame.cols);
        int centerY = (int)(data[1] * frame.rows);
        int width = (int)(data[2] * frame.cols);
        int height = (int)(data[3] * frame.rows);
        int left = centerX - width / 2;
        int top = centerY - height / 2;

        classIds.push_back(classIdPoint.x);
        confidences.push_back((float)confidence);
        boxes.push_back(Rect(left, top, width, height));
        centers.push_back(Point2d(centerX, centerY));
        float depth = out_image_depth.at<float>(centerY, centerX);
        if (depth < 3.5){
          
          if (!(near_objects.find(classes[classIdPoint.x]) != near_objects.end())){
            std::cout << "Objeto detectado: " << classes[classIdPoint.x] << " a " << depth << "m\n";
            near_objects_distance.insert(depth);
          }
          near_objects.insert(classes[classIdPoint.x]);
        }
        else{
          far_objects.insert(classes[classIdPoint.x]);
        }
      }
    }
    
  }

  // Perform non maximum suppression to eliminate redundant overlapping boxes with
  // lower confidences
  vector<int> indices;
  NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
  for (size_t i = 0; i < indices.size(); ++i) {
    int idx = indices[i];
    Rect box = boxes[idx];
    drawPred(
      classIds[idx], confidences[idx], box.x, box.y,
      box.x + box.width, box.y + box.height, frame, centers[idx], out_image_depth);
  }
}

// Draw the predicted bounding box


// Get the names of the output layers
vector<String> getOutputsNames(const Net & net)
{
  static vector<String> names;
  if (names.empty()) {
    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    vector<int> outLayers = net.getUnconnectedOutLayers();

    //get the names of all the layers in the network
    vector<String> layersNames = net.getLayerNames();

    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) {
      names[i] = layersNames[outLayers[i] - 1];
    }
  }
  return names;
}

// <------- OPCION 4 --------/>

void calcDirection(int R, int L, int F, int B){
  if (R > L && R > F && R > B){
    std::cout << "Derecha\n";
    contRIGHT++;
    
    if (contRIGHT >= 12){
      if (direction == RIGHT){
        direction = DOWN;
      }
      else if (direction == LEFT){
        direction = UP;
      }
      else if (direction == UP){
        direction = RIGHT;
      }
      else if (direction == DOWN){
        direction = LEFT;
      }
      
      contRIGHT = 0;
      contDOWN = 0;
      contUP = 0;
      contLEFT = 0;
    }

  }
  else if (L > R && L > F && L > B){
    std::cout << "Izquierda\n";
    contLEFT++;

    if (contLEFT >= 12){
      if (direction == RIGHT){
        direction = UP;
      }
      else if (direction == LEFT){
        direction = DOWN;
      }
      else if (direction == UP){
        direction = LEFT;
      }
      else if (direction == DOWN){
        direction = RIGHT;
      }
      contRIGHT = 0;
      contDOWN = 0;
      contUP = 0;
      contLEFT = 0;
    }
  }
  // else if (F > R && F > L && F > B){
  //   std::cout << "Adelante\n";
    
  //   contUP++;
  //   if (contUP > 100){
  //     if (direction == RIGHT){
  //       position.y += 1;
  //       mapa.at<uchar>(position.x, position.y) = 255;
  //     }
  //     else if (direction == LEFT){
  //       position.y -= 1;
  //       mapa.at<uchar>(position.x, position.y) = 255;
  //     }
  //     else if (direction == UP){
  //       position.x -= 1;
  //       mapa.at<uchar>(position.x, position.y) = 255;
  //     }
  //     else if (direction == DOWN){
  //       position.x += 1;
  //       mapa.at<uchar>(position.x, position.y) = 255;
  //     }
  //   }


  // }
  else if (B > R && B > L && B > F){
    std::cout << "Atras\n";
    contDOWN++;
    if (contDOWN > 10){
      contDOWN = 0;
      contUP = 0;
      contRIGHT = 0;
      contLEFT = 0;
      if (direction == RIGHT){
        position.y -= 1;
        mapa.at<uchar>(position.x, position.y) = 255;
      }
      else if (direction == LEFT){
        position.y += 1;
        mapa.at<uchar>(position.x, position.y) = 255;
      }
      else if (direction == UP){
        position.x += 1;
        mapa.at<uchar>(position.x, position.y) = 255;
      }
      else if (direction == DOWN){
        position.x -= 1;
        mapa.at<uchar>(position.x, position.y) = 255;
      }
    }

  }
}



void localitation(){
  std::vector<std::set<std::string>> detected_objects;
  

  detected_objects.push_back(near_objects);
  detected_objects.push_back(far_objects);

  int x = 0;
  for (room it : rooms){
    
    int room_counter = 0;
     
    for (string obj : near_objects){
      if(std::find(it[0].begin(), it[0].end(), obj) != it[0].end()){
        room_counter++;
        std::cout << "Objeto cercano coincide: " << obj << "\n";
      }
    }
    for (string obj : far_objects){
      if(std::find(it[1].begin(), it[1].end(), obj) != it[1].end()){
        room_counter++;
        std::cout << "Objeto lejano coincide: " << obj << "\n";
      }
    }
    std::cout << "Habitacion "<< room_names[x] << " coincidencias: " << to_string(room_counter) << "\n";
    std::cout<<std::endl;
    double percentage = (double)room_counter / (double)(it[0].size() + it[1].size());
    conc_perc.push_back(percentage);
    x++;
  }

  // std::cout << "Objetos detectados cerca: ";
  // if (!near_objects.empty()){
  //   int i = 0;
  //   for (auto it = near_objects.begin(); it != near_objects.end(); ++it, i++){

  //     std::cout << i << " ";
  //     std::cout << *it << " ";
      
  //   }
  //   for (auto it = near_objects_distance.begin(); it != near_objects_distance.end(); ++it){
  //     std::cout << *it << "m ";
  //   }
  // }
  // std::cout << "\n";
  // std::cout << "Objetos detectados lejos: ";
  // if(!far_objects.empty()){
  //   for (auto it = far_objects.begin(); it != far_objects.end(); ++it){
  //     std::cout << *it << " ";
  //   }
  // }
  auto max_perc = std::max_element(conc_perc.begin(), conc_perc.end());
  size_t i = std::distance(conc_perc.begin(), max_perc);

  std::cout << "La habitacion mas probable es: " << room_names[i]<<" con indice "<<to_string(i) << "\n";
  near_objects.clear();
  far_objects.clear();
  detected_objects.clear();
  conc_perc.clear();


}



namespace computer_vision
{
void CVSubscriber::topic_callback_vel(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  if (option == 3){
    if (msg->linear.x > 0.1){
      cout<< "Linear x"<<  to_string(msg->linear.x) << std::endl;
      contRIGHT = 0;
      contDOWN = 0;
      contUP = 0;
      contLEFT = 0;
      if (direction == RIGHT){
        position.y += 1;
        mapa.at<uchar>(position.x, position.y) = 255;
      }
      else if (direction == LEFT){
        position.y -= 1;
        mapa.at<uchar>(position.x, position.y) = 255;
      }
      else if (direction == UP){
        position.x -= 1;
        mapa.at<uchar>(position.x, position.y) = 255;
      }
      else if (direction == DOWN){
        position.x += 1;
        mapa.at<uchar>(position.x, position.y) = 255;
      }
    }
  }
}

void CVSubscriber::topic_callback_key(const std_msgs::msg::String::SharedPtr msg)
{
  if (option == 4){
    if (msg->data == "l"){
      turn_flag = true;
   
    }
  }
}
  
/**
   TO-DO: Default - the output images and pointcloud are the same as the input
 */
CVGroup CVSubscriber::processing(
  const cv::Mat in_image_rgb,
  const cv::Mat in_image_depth,
  const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
const
{
  cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
  if (first){
    cv::createTrackbar("Option[0-4]", winname, nullptr, 4, 0);
    cv::setTrackbarPos("Option[0-4]", winname, 0);
    first = false;
    
  }

  // Create output images
  cv::Mat out_image_rgb, out_image_depth;
  // Create output pointcloud
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud;

  // Processing
  out_image_rgb = in_image_rgb;
  out_image_depth = in_image_depth;
  out_pointcloud = in_pointcloud;

  option = cv::getTrackbarPos("Option[0-4]", winname);

  if (option == 0){
    imshow(winname, out_image_rgb);
    
  }else if (option == 1){
    cv::Mat reconstruct = cv::Mat::zeros(in_image_rgb.rows,in_image_rgb.cols, CV_8UC3);
    std::vector<cv::Point3d> pts;
    std::vector<cv::Scalar> pts_color;
    for (int i = 0; i < (int)out_pointcloud.size(); i++){
      cv::Point3d pt(out_pointcloud.points[i].x, out_pointcloud.points[i].y, out_pointcloud.points[i].z);
      cv::Scalar color(out_pointcloud.points[i].b, out_pointcloud.points[i].g, out_pointcloud.points[i].r);
      pts.push_back(pt);
      pts_color.push_back(color);
    }
    cv::Matx33d camera_matrix = camera_model_->fullIntrinsicMatrix();
    
    // matriz de rotacion nula 
    // Inicializar la matriz de rotación
    cv::Mat rotation_matrix = cv::Mat::eye(3, 3, CV_64F); 

    std::vector<cv::Point2d> pt_out;
    
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0,0,0);

    projectPoints(pts, rotation_matrix, tvec, camera_matrix, cv::noArray(), pt_out);

    for (int i = 0; i < (int)pt_out.size(); i++){
      cv::Point2d pt = pt_out[i];
      if (pt.x >= 0 && pt.x < in_image_rgb.cols && pt.y >= 0 && pt.y < in_image_rgb.rows){
      reconstruct.at<cv::Vec3b>(pt.y, pt.x) =  cv::Vec3b(pts_color[i][0], pts_color[i][1], pts_color[i][2]);
      }
    }
    cv::Mat suavizada = reconstruct.clone();

    for( int i = 0; i < (int)reconstruct.rows; i++ ) {
      for( int j = 0; j < (int)reconstruct.cols; j++ ) {
        
        if (reconstruct.at<cv::Vec3b>(i, j) == cv::Vec3b(0,0,0)){
          std::vector<cv::Vec3b> colors;
          
          // mirar si hay algun pixel vecino que no sea negro
          for(int z = -1; z <= 1; z++){
            for(int x = -1; x <= 1; x++){
              if (i + z <0 || i + z >= reconstruct.rows || j + x < 0 || j + x >= reconstruct.cols){
                continue;
              }
              if (reconstruct.at<cv::Vec3b>(i + z, j + x) != cv::Vec3b(0,0,0) ){
                colors.push_back(reconstruct.at<cv::Vec3b>(i + z, j + x ));
                // new_color = reconstruct.at<cv::Vec3b>(i + z, j + x );
                break;
              }
            }
          }
          cv::Mat colorsMat(colors);
          cv::Scalar media = cv::mean(colorsMat);
          cv::Vec3b colorMed = cv::Vec3b(media[0],media[1],media[2]);

          suavizada.at<cv::Vec3b>(i, j) = colorMed;
        }
        else{
          suavizada.at<cv::Vec3b>(i, j) = reconstruct.at<cv::Vec3b>(i, j);
        }
        
      }
      
    }
    
    imshow(winname, suavizada);
  
 
  }
  else if (option == 2){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr image_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    cv::Matx33d camera_matrix = camera_model_->fullIntrinsicMatrix();

    double cx = camera_matrix(0, 2);
    double cy = camera_matrix(1, 2);
    double fx = camera_matrix(0, 0);
    double fy = camera_matrix(1, 1);
    
    for (int i = 0; i < (int)out_image_rgb.rows; i++){
      for (int j = 0; j < (int)out_image_rgb.cols; j++){
        float d = out_image_depth.at<float>(i, j);

        double X = ((j - cx) * d) / fx;
        double Y = ((i - cy) * d) / fy;
        double Z = d;

        cv::Mat pts3d_transformed = (cv::Mat_<double>(4, 1) << X, Y, Z, 1);

        double x = pts3d_transformed.at<double>(0);
        double y = pts3d_transformed.at<double>(1);
        double z = pts3d_transformed.at<double>(2);

        // coger el color de cada punto
        cv::Vec3b color = out_image_rgb.at<cv::Vec3b>(i, j);
        image_cloud->points.push_back(pcl::PointXYZRGB(x, y, z, color[2], color[1], color[0]));
      }
    }
    
    out_pointcloud = *image_cloud;
  }

  else if (option == 3){
      // Create some random colors
    if (colors.empty()) {
      // std::cout << "Buscando puntos de interes\n";
      RNG rng;
      for (int i = 0; i < 100; i++) {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r, g, b));
      }
      mapa = Mat::zeros(600, 600, CV_8UC1);
      mapa.at<uchar>(300, 300) = 255;

    }
    

    // Take first frame and find corners in it
    if(p0.size() < 4){
      p0.clear();
      p1.clear();
      old_frame = in_image_rgb.clone();
      cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
      goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

      // Create a mask image for drawing purposes
      mask = Mat::zeros(old_frame.size(), old_frame.type());
      flag_recalc = 0;
    }
    
    
    Mat frame, frame_gray;

    frame = in_image_rgb.clone();
    
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    // std::cout << "Antes del calculo\n";
    // calculate optical flow
    vector<uchar> status;
    vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) +(TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 5, criteria);

    vector<Point2f> good_new;
    // std::cout << "Despues del calculo\n";
    int R = 0;
    int L = 0;
    int F = 0;
    int B = 0;




    
    // int point_statics = 0;
    for (uint i = 0; i < p0.size(); i++) {
      // Select good points
      
      if (status[i] == 1) {
        

        good_new.push_back(p1[i]);


        
        // draw the tracks
        line(mask, p1[i], p0[i], colors[i], 2);
        circle(frame, p1[i], 5, colors[i], -1);
        
        if (abs(p1[i].x - p0[i].x) < 0.05 || abs(p1[i].y - p0[i].y) < 0.05)
          {
            // point_statics++;
            continue;
          }
          
        
        if (p1[i].x- p0[i].x > 20)
        {
          L++;
        }
        else if (p1[i].x - p0[i].x < -20)
        {
          R++;
        }
        
        if(p1[i].x > out_image_rgb.cols/2){
          
          
          double pendiente = (p1[i].y - p0[i].y) / (p1[i].x - p0[i].x);
          if (isnan(pendiente) || isinf(pendiente)){
            continue;
          }
          if (pendiente > 0.5 && p1[i].y < out_image_rgb.rows/2 )
          {
            F++;
          }
          else if (pendiente < -0.5 && p1[i].y > out_image_rgb.rows/2)
          {
            F++;
          }
          

        }
      }
    }

    // if (point_statics > (int)p0.size()/2){
    //   flag_recalc = 1;
      
    // }
      
      

    calcDirection(R, L, F, B);
    
    Mat img;
    add(frame, mask, img);

    imshow("Frame", img);

    imshow("Mapa", mapa);
    // Now update the previous frame and previous points
    old_gray = frame_gray.clone();
    p0 = good_new;
    
  }
  else if (option == 4){
    if (classes.empty()){


      // Load names of classes
      std::filesystem::path path_absoluto = std::filesystem::absolute("src/practicafinal-grupo15/src/practicafinal-grupo15/cfg/coco.names");
      string classesFile = path_absoluto.string();
      ifstream ifs(classesFile.c_str());
      string line;
      while (getline(ifs, line)) {classes.push_back(line);}

      string device = "gpu";
      

      // Give the configuration and weight files for the model
      std::filesystem::path model_conf_path_absoluto = std::filesystem::absolute("src/practicafinal-grupo15/src/practicafinal-grupo15/cfg/yolov3.cfg");
      std::filesystem::path model_weights_path_absoluto = std::filesystem::absolute("src/practicafinal-grupo15/src/practicafinal-grupo15/cfg/yolov3.weights");
      string modelConfiguration = model_conf_path_absoluto.string();
      string modelWeights = model_weights_path_absoluto.string();

      // Load the network
      net = readNetFromDarknet(modelConfiguration, modelWeights);

      if (device == "cpu") {
        cout << "Using CPU device" << endl;
        net.setPreferableBackend(DNN_TARGET_CPU);
      } else if (device == "gpu") {
        cout << "Using GPU device" << endl;
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
      }
      namedWindow(kWinName, WINDOW_NORMAL);
      // empezamos a contar tiempo 
      start_time = std::clock();
    }

    if(!turn_flag){
      if (first_time){
        
        first_time = false;

        // Guardar la configuración actual de la terminal
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;

        // Modify the settings
        newt.c_lflag &= ~ICANON; // Disable canonical mode
        newt.c_lflag &= ~ECHO; // Disable echo
        newt.c_cc[VMIN] = 0; // Minimum input characters = 0
        newt.c_cc[VTIME] = 0; // Time to wait for input = 0

        // Aplicar la nueva configuración
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        // Establecer la entrada estándar en modo no bloqueante
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
      }
      ch = '\0';
      // std::cout << "Esperando input\n";
      if (read(STDIN_FILENO, &ch, 1) < 0)
        {
            // Error reading from stdin
            if (errno != EAGAIN){
            perror("read");
            }
        }
      else{
        if (ch == 'l')
        {
          std::cout << "Girando\n";
          turn_flag = true;
          first_time = true;
          tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
            // Do something when 'l' is pressed
        }
      }

      

        
        
    } else{
      // do navega while no estes en waypoint

      // giramos y yolo para detectar objetos
      // si el tiempo de giro acaba decision localizacion con listas de objetos encontrados 
      // publicar giro
      auto msg = std::make_unique<geometry_msgs::msg::Twist>();
      
      msg->angular.z = 0.7;
      publisher_vel_->publish(std::move(msg));
      
      cv::Mat frame, blob;
      frame = in_image_rgb.clone();
      
      // Create a 4D blob from a frame.
      blobFromImage(
        frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(
          0, 0,
          0), true, false);

      //Sets the input to the network
      net.setInput(blob);

      // Runs the forward pass to get output of the output layers
      vector<Mat> outs;
      net.forward(outs, getOutputsNames(net));

      // Remove the bounding boxes with low confidence
      postprocess(frame, outs, out_image_depth);


      // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
      vector<double> layersTimes;
      double freq = getTickFrequency() / 1000;
      double t = net.getPerfProfile(layersTimes) / freq;
      string label = format("Inference time for a frame : %.2f ms", t);
      putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

      // Write the frame with the detection boxes
      Mat detectedFrame;
      frame.convertTo(detectedFrame, CV_8U);

      std::clock_t end_time = std::clock();
      double elapsed_seconds = static_cast<double>(end_time - start_time) / (CLOCKS_PER_SEC*10) ;

      if (elapsed_seconds > 14){
        // std::cout << "Tiempo de giro acabado t = "<< elapsed_seconds << "s" << "\n";
        // publicar stop
        localitation();
        start_time = std::clock();
        turn_flag = false;
      }
      imshow(kWinName, frame);
    }
    imshow(winname, out_image_rgb);


  }
  else {
    imshow(winname, out_image_rgb);
  }
  // Show images in a different windows
  
  cv::waitKey(3);

  return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
}

} // namespace computer_vision

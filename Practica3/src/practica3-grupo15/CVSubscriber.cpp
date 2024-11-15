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

#include "computer_vision/CVSubscriber.hpp"


using namespace cv;
using namespace std;

bool first = true;
string winname = "Practice3";
int sx = -1, sy = -1, option3 = 0;
std::vector <cv::Point2d> points2d;
// create mouse callback
void on_mouse(int event, int x, int y, int, void*)
{
  switch (event){
    case cv::EVENT_LBUTTONDOWN:
      if (option3 == 1){
        sx = x;
        sy = y;
        points2d.push_back(cv::Point2d(x, y));
      }
      break;
    default:

      break;
  }
}

void line_detection(cv::Mat &hsv, cv::Mat &mask){
      // //  0 0 0 170 10 255
    cv::inRange(hsv, cv::Scalar(26, 115, 107), cv::Scalar(60,255,255), mask);
  
    // Contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<Moments> mu(contours.size() );
    vector<Point2f> centroids(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(mask, contours, i,Scalar(255,255,255), 6, LINE_8, hierarchy, 0);
    }

    vector<vector<Point>> contours2;
    vector<Vec4i> hierarchy2;
    findContours(mask, contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<Moments> mu2(contours.size() );
    vector<Point2f> centroids2(contours.size());
    

    for (size_t i = 0; i < contours2.size(); i++) {
        drawContours(mask, contours2, i,Scalar(255,255,255), 5, LINE_8, hierarchy2, 0);
    }
  
}

 pcl::PointCloud<pcl::PointXYZRGB>::Ptr extract_planes(pcl::PCLPointCloud2::Ptr cloud_blob){
    
  pcl::PCLPointCloud2::Ptr  cloud_filtered_blob (new pcl::PCLPointCloud2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor2;
  sor2.setInputCloud (cloud_blob);
  sor2.setLeafSize (0.01f, 0.01f, 0.01f);
  sor2.filter (*cloud_filtered_blob);

  // Convert to the templated PointCloud
  pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;

  // Optional
  seg.setOptimizeCoefficients (true);

  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (1000);
  seg.setDistanceThreshold (0.01);

  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  int i = 0, nr_points = (int) cloud_filtered->size ();
  // While 10% of the original cloud is still there
  while (cloud_filtered->size () > 0.1 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    Eigen::Vector4f min_pt, max_pt;
    pcl::Indices indices = std::vector<int>(inliers->indices);
    //obtencion maximo y minimo de la nube de puntos filtarada con indices de inliers
    pcl::getMinMax3D(*cloud_filtered, indices, min_pt, max_pt);

    double A = coefficients->values[0];
    double B = coefficients->values[1];
    double C = coefficients->values[2];
    double D = coefficients->values[3];

    if (std::abs(B) > std::abs(C) && std::abs(B) > std::abs(A)){
      for (double i = min_pt[0]; i <= max_pt[0]; i += 0.01) {  
          for (double j = min_pt[2]; j <= max_pt[2]; j += 0.01) {
              double x = i;
              double z = j;
              double y = -(D + C * z + A * x ) / B;
              plane_cloud->points.push_back(pcl::PointXYZRGB(x, y, z, 255, 0, 0));
          }
      }
    }

    if (inliers->indices.size () == 0)
    {
      std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the inliers
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_p);

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered.swap (cloud_f);
    i++;
  }

  return plane_cloud;
}



pcl::PointCloud<pcl::PointXYZRGB> draw_cubes(geometry_msgs::msg::TransformStamped transform){
  int distance = getTrackbarPos("Distance[0-8]", winname);
  double cube_size = 0.08;

  pcl::PointCloud<pcl::PointXYZRGB> cubes;
  std::vector<cv::Point3d> pts;
  
  cv::Mat tvec = (cv::Mat_<double>(3, 1) << transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);
  for (double n = 0.0; n <= distance; n++){
      pts.push_back(cv::Point3d( 0.45, tvec.at<double>(1,0), n));
      pts.push_back(cv::Point3d(-0.45, tvec.at<double>(1,0), n));
    }

  int green = 255;
  int red = 0;
  for (int i = 0; i < (int)pts.size(); i++){
    if (i % 2 == 0){
      if ((int)(i+1) < 6){
        green -= 64;
      }
      else if ((int)(i+1) > 6 && (int)(i+1) <= 8 ){
        red = 127;
        green = 80;
      }
      else if ((int)(i+1) > 8 && (int)(i+1) <= 10 ){
        red = 127;
        green = 0;
      }
      else{
        red += 25;
      }
      
    }
    for (double j = pts[i].x ; j < pts[i].x + cube_size; j += 0.01){
      for (double k = pts[i].y ; k > pts[i].y - cube_size; k -= 0.01){
        for (double l = pts[i].z ; l < pts[i].z + cube_size; l += 0.01){

          cubes.push_back(pcl::PointXYZRGB(j, k, l, red, green, 0));
        }
      }
    }
 } 
 return cubes;
}
 

namespace computer_vision
{

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


    cv::createTrackbar("Iterations[0-20]", winname, nullptr, 20, 0);
    cv::setTrackbarPos("Iterations[0-20]", winname, 0);


    cv::createTrackbar("Distance[0-8]", winname, nullptr, 8, 0);
    cv::setTrackbarPos("Distance[0-8]", winname, 0);

    // create mouse callback
    cv::setMouseCallback( winname, on_mouse, 0 );
    
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

  int option = cv::getTrackbarPos("Option[0-4]", winname);

  if (option != 3){
    points2d.clear();
    option3 = 0;
  }

  if (option == 0){
    imshow(winname, out_image_rgb);
    // borrar puntos una vez los hayamos hecho
    
  }else if (option == 1){
    cv::Mat gray, canny, hsv, mask,InvMask, mask2,InvMask2, filtered;
    
    cv::cvtColor(in_image_rgb, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(in_image_rgb, hsv, cv::COLOR_BGR2HSV);

    line_detection(hsv, mask);

    for (int i = 0; i < out_image_rgb.cols; i++) {
      for (int j = 0; j < out_image_rgb.rows; j++) {
        Scalar intensity = mask.at<uchar>(j, i);
        if (intensity.val[0] == 255) {
          out_image_rgb.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
        }
      }
    }
    cv::Mat skeleton = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::Mat open, temp, eroded;
    int iterations = cv::getTrackbarPos("Iterations[0-20]", winname);

    for (int i = 0; i < iterations; i++){
      cv::morphologyEx(mask, open, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
      temp = mask - open;
      cv::erode(mask, eroded, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
      skeleton = skeleton | temp;  
      mask = eroded;
    }
    Mat suavizada;
    GaussianBlur(skeleton, suavizada, Size(5,5), 0, 0);
    
    // imshow("Skeleton", skeleton);
    out_image_rgb = in_image_rgb.clone();

    for (int i = 0; i < out_image_rgb.cols; i++) {
      for (int j = 0; j < out_image_rgb.rows; j++) {
        Scalar intensity = suavizada.at<uchar>(j, i);
        if (intensity.val[0] > 50) {
          out_image_rgb.at<Vec3b>(j, i) = Vec3b(0, 0, 255);
        }
      }
    }

    imshow(winname, out_image_rgb);

  }else if (option == 2){
    // // Se obtiene la matriz de parámetros intrínsecos de la cámara.
    cv::Matx33d camera_matrix = camera_model_->fullIntrinsicMatrix();

    double distance = cv::getTrackbarPos("Distance[0-8]", winname);

    
    tf2::Quaternion quaternion(transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w);
    
    tf2::Matrix3x3 mat(quaternion);
    
    // Convert tf2::Matrix3x3 to cv::Mat
    cv::Mat rotation_matrix(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
          rotation_matrix.at<double>(i, j) = mat[i][j];
      }
    }
  
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z);
    
    std::vector<cv::Point3d> pts;
    std::vector<cv::Point2d> pt_out;

    for (double n = 0.0; n <= distance; n++){
      pts.push_back(cv::Point3d(n, 0.45, 0.0));
      pts.push_back(cv::Point3d(n, 0.0, 0.0));
      pts.push_back(cv::Point3d(n, -0.45, 0.0));
    }

    projectPoints(pts, rotation_matrix, tvec, camera_matrix, cv::noArray(), pt_out);

    int green = 255;
    int red = 0;
    for (int i = 0; i < (int)pts.size(); i++){
      if (i % 3 == 0){
        if ((int)(i+1) < 12){
          green -= 15;
        } else if ((int)(i+1) > 11 && (int)(i+1) <= 14 ){
          green = 127;
        }
        else if ((int)(i+1) > 14 && (int)(i+1) <= 17 ){
          red = 127;
          green = 80;
        }
        else if((int)(i+1) > 17 && (int)(i+1) <= 26 ){
          red += 42;
          green = 0;
        }
      }

      cv::circle(out_image_rgb, pt_out[i], 5, cv::Scalar(0, green, red), -1);
      if ((int)(i+1) % 3 == 0){
        putText(out_image_rgb, std::to_string(((int)(i+1) / 3)-1), Point(pt_out[i].x + 5, pt_out[i].y), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, green, red), 1);
      }
    }
    
    cv::imshow(winname, out_image_rgb);

  }else if (option == 3){
    option3 = 1;
    // // Se obtiene la matriz de parámetros intrínsecos de la cámara.
    cv::Matx33d camera_matrix = camera_model_->fullIntrinsicMatrix();

    // 2d a 3d    
    double cx = camera_matrix(0, 2);
    double cy = camera_matrix(1, 2);
    double fx = camera_matrix(0, 0);
    double fy = camera_matrix(1, 1);

    tf2::Quaternion quaternion(transform2.transform.rotation.x, transform2.transform.rotation.y, transform2.transform.rotation.z, transform2.transform.rotation.w);
    
    tf2::Matrix3x3 mat(quaternion);
    
    // Convert tf2::Matrix3x3 to cv::Mat
    cv::Mat rotation_matrix(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
          rotation_matrix.at<double>(i, j) = mat[i][j];
      }
    }

    cv::Mat tvec = (cv::Mat_<double>(1, 3) << transform2.transform.translation.x, transform2.transform.translation.y, transform2.transform.translation.z);

    double factor = pow(10, 3);

    cv::Matx34d extrinsic_matrix = cv::Matx34d(round(rotation_matrix.at<double>(0, 0)), round(rotation_matrix.at<double>(0, 1)), round(rotation_matrix.at<double>(0, 2)), round(tvec.at<double>(0) * factor)/factor,
                                    round(rotation_matrix.at<double>(1, 0)), round(rotation_matrix.at<double>(1, 1)), round(rotation_matrix.at<double>(1, 2)), round(tvec.at<double>(1) * factor)/factor,
                                    round(rotation_matrix.at<double>(2, 0)), round(rotation_matrix.at<double>(2, 1)), round(rotation_matrix.at<double>(2, 2)), round(tvec.at<double>(2) * factor)/factor);

    
    for (int i = 0; i < (int)points2d.size(); i++){
      float d = out_image_depth.at<float>(points2d[i].y, points2d[i].x);

      if (std::isnan(d)){
        cv::circle(out_image_rgb, points2d[i], 5, cv::Scalar(0, 0, 255), -1);
        putText(out_image_rgb, "[inf, inf, -inf]", Point(points2d[i].x + 8, points2d[i].y), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
      } else {
        double X = ((points2d[i].x - cx) * d) / fx;
        double Y = ((points2d[i].y - cy) * d) / fy;
        double Z = d;

        cv::Mat pts3d_transformed = (cv::Mat_<double>(4, 1) << X, Y, Z, 1);
        cv::Mat point3d_transformed = extrinsic_matrix * pts3d_transformed;

        cv::circle(out_image_rgb, points2d[i], 5, cv::Scalar(0, 0, 255), -1);
        std::ostringstream text;
        text << "[" << std::fixed << std::setprecision(2) << round(point3d_transformed.at<double>(0,0) * 100) / 100 << ", "
            << std::fixed << std::setprecision(2) << round(point3d_transformed.at<double>(0,1) * 100) / 100 << ", "
            << std::fixed << std::setprecision(2) << round(point3d_transformed.at<double>(0,2) * 100) / 100 << "]";
        putText(out_image_rgb, text.str(), Point(points2d[i].x + 8, points2d[i].y), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
      }
    }
    

    imshow(winname, out_image_rgb);
  }else if (option == 4){
    // int Low_H = cv::getTrackbarPos("Low_H", winname);
    // int Low_S = cv::getTrackbarPos("Low_S", winname);
    // int Low_V = cv::getTrackbarPos("Low_V", winname);
    // int High_H = cv::getTrackbarPos("High_H", winname);
    // int High_S = cv::getTrackbarPos("High_S", winname);
    // int High_V = cv::getTrackbarPos("High_V", winname);
    
    pcl::PointCloud<pcl::PointXYZHSV>::Ptr hsvCloud(new pcl::PointCloud<pcl::PointXYZHSV>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
 
    
    pcl::PointCloudXYZRGBtoXYZHSV(in_pointcloud, *hsvCloud);
  

    pcl::PointCloud<pcl::PointXYZRGB> new_pointcloud;
    //filtro de color
    //50 90 0 70 110 130
    for (int i = 0; i < (int)hsvCloud->size(); i++){
      pcl::PointXYZHSV point = hsvCloud->at(i);
      pcl::PointXYZRGB pointRGBf;
      if (point.h > 50 && point.h  < 70 && point.s * 100 > 90 && point.s  * 100< 110 && point.v * 100 > 0 && point.v * 100 < 130){
        pointRGBf.x = point.x;
        pointRGBf.y = point.y;
        pointRGBf.z = point.z;
        pointRGBf.r = 255;
        pointRGBf.g = 0;
        pointRGBf.b = 0;
        new_pointcloud.push_back(pointRGBf);
      }

    }
    out_pointcloud = new_pointcloud;

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud (out_pointcloud.makeShared());
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*filteredCloud);

    out_pointcloud = *filteredCloud;
    pcl::PCLPointCloud2::Ptr pcl_pc2(new pcl::PCLPointCloud2);;
    pcl::toPCLPointCloud2(out_pointcloud, *pcl_pc2);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    plane_cloud = extract_planes(pcl_pc2);

 


    out_pointcloud = *plane_cloud;

    pcl::PointCloud<pcl::PointXYZRGB> cubes = draw_cubes(transform);
    
    out_pointcloud += cubes;

    cv::imshow(winname, out_image_rgb);
  }else {
    imshow(winname, out_image_rgb);
  }
    



  // Show images in a different windows
  
  cv::waitKey(3);

  return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
}

} // namespace computer_vision

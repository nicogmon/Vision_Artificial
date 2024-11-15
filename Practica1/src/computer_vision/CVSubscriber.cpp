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
bool first = true;
int pulse = 0;

cv::Mat fftShift( cv::Mat & magI)
  {
    cv::Mat magI_copy = magI.clone();
    // crop the spectrum, if it has an odd number of rows or columns
    magI_copy = magI_copy(cv::Rect(0, 0, magI_copy.cols & -2, magI_copy.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI_copy.cols / 2;
    int cy = magI_copy.rows / 2;

    cv::Mat q0(magI_copy, cv::Rect(0, 0, cx, cy));     // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI_copy, cv::Rect(cx, 0, cx, cy));    // Top-Right
    cv::Mat q2(magI_copy, cv::Rect(0, cy, cx, cy));    // Bottom-Left
    cv::Mat q3(magI_copy, cv::Rect(cx, cy, cx, cy));   // Bottom-Right

    cv::Mat tmp;                             // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                      // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    return magI_copy;
  }

cv::Mat computeDFT( cv::Mat & image)
  {
    // Expand the image to an optimal size.
    cv::Mat padded1;

    int m1 = cv::getOptimalDFTSize(image.rows);
    int n1 = cv::getOptimalDFTSize(image.cols);
    cv::copyMakeBorder(image, padded1, 0, m1 - image.rows, 0, n1 - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  
    cv::Mat planes[] = {cv::Mat_<float>(padded1), cv::Mat::zeros(padded1.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    cv::dft(complexI, complexI, cv::DFT_COMPLEX_OUTPUT);

    return complexI;
  }

cv::Mat spectrum( cv::Mat & complexI)
  {
    cv::Mat complexImg = complexI.clone();
    cv::Mat shift_complex = fftShift(complexImg);

    cv::Mat planes_spectrum[2];

    cv::split(shift_complex, planes_spectrum);

    cv::magnitude(planes_spectrum[0], planes_spectrum[1], planes_spectrum[0]);
    cv::Mat spectrum = planes_spectrum[0];

    spectrum += cv::Scalar::all(1);
    cv::log(spectrum, spectrum);

    cv::normalize(spectrum, spectrum, 0, 1, cv::NORM_MINMAX);

    // std::cout<<"spectrum type: "<<spectrum.type()<<std::endl;

    return spectrum;
  }


cv::Mat option4(cv::Mat out_image_rgb){
  cv::cvtColor(out_image_rgb, out_image_rgb, cv::COLOR_BGR2GRAY);
  cv::Mat complexImg = computeDFT(out_image_rgb);
  cv::Mat spectrum_original = spectrum(complexImg);
  cv::Mat shift_complex = fftShift(complexImg);
    
    
  int cy = complexImg.rows / 2;
    
  int filterSize = cv::getTrackbarPos("Filter value[0-100]", "Practice1");
  if (filterSize == 0){
    cv::Mat inverseTransform;
    idft(shift_complex, inverseTransform, cv::DFT_REAL_OUTPUT );
    cv::normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);
    return inverseTransform;
  }
   
  //std::cout<<"shift.complex typpe"<<shift_complex.type()<<std::endl;

  cv::Mat filter = cv::Mat::zeros(shift_complex.size(), shift_complex.type());
  

  cv::line(filter, cv::Point(0,  cy), cv::Point(filter.cols, cy), cv::Scalar(1,1,1),  filterSize);

  mulSpectrums(filter, shift_complex, shift_complex, 0);
  cv::Mat rearrange = fftShift(shift_complex);
  
  cv::Mat inverseTransform;
  idft(rearrange, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
  cv::normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);
  
  //return inverseTransform;
  return inverseTransform;
}

cv::Mat option5(cv::Mat out_image_rgb){
  cv::cvtColor(out_image_rgb, out_image_rgb, cv::COLOR_BGR2GRAY);
  cv::Mat complexImg = computeDFT(out_image_rgb);
  cv::Mat spectrum_original = spectrum(complexImg);
  cv::Mat shift_complex = fftShift(complexImg);
    
    
  int cy = complexImg.rows / 2;
    
  int filterSize = cv::getTrackbarPos("Filter value[0-100]", "Practice1");
  if (filterSize == 0){
    cv::Mat inverseTransform;
    idft(shift_complex, inverseTransform, cv::DFT_REAL_OUTPUT );
    cv::normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);
    return inverseTransform;
  }
   
  //std::cout<<"shift.complex typpe"<<shift_complex.type()<<std::endl;

  cv::Mat filter = cv::Mat::ones(shift_complex.size(), shift_complex.type());
  

  cv::line(filter, cv::Point(0,  cy), cv::Point(filter.cols, cy), cv::Scalar(0,0,0),  filterSize);

  mulSpectrums(filter, shift_complex, shift_complex, 0);
  cv::Mat rearrange = fftShift(shift_complex);
  
  cv::Mat inverseTransform;
  idft(rearrange, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
  cv::normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);
  
  //return inverseTransform;
  return inverseTransform;

}

cv::Mat rgb_to_hsi(cv::Mat rgb_image) {
    // std::cout<<"pixel[0]: "<<(uint)pixel0<<std::endl;
    // std::cout<<"pixel[1]: "<<(uint)pixel1<<std::endl;
    // std::cout<<"pixel[2]: "<<(uint)pixel2<<std::endl;
    cv::Mat hsi_image = rgb_image.clone();
    for (int i = 0; i < hsi_image.rows; i++){
      for (int j = 0; j < hsi_image.cols; j++){
        cv::Vec3b pixel = hsi_image.at<cv::Vec3b>(i, j);
        double r =  (pixel[2] / 255.0);
        double g =  (pixel[1] / 255.0);
        double b =  (pixel[0] / 255.0);

        double intensity = (r + g + b) / 3.0;

        double min_val = std::min({r, g, b});
        double saturation;
        
        
        saturation = 1.0 - 3.0 *( min_val / (r + g + b)+0.00000001);
        
        double hue;
        
        hue = (180.0/M_PI) * acos(0.5 * ((r - g) + (r - b)) / (sqrt(pow((r - g), 2) + ((r - b) * (g - b)))+0.00000001));
        
        // std::cout<<"theta: "<<theta<<std::endl;

        
        if (b > g){
            hue = 360.0 - hue;
        }

        
        double Hue =  abs(((hue/360.0) * 255.0));
        double Saturation = abs(saturation * 255.0);
        double Intensity = abs(intensity * 255.0);
        hsi_image.at<cv::Vec3b>(i, j) =  cv::Vec3b(Hue, Saturation, Intensity);
        
      }
    }
    

    // std::cout<<"hsi_pixel[0]: "<<(uint)hsi_pixel[0]<<std::endl;
    // std::cout<<"hsi_pixel[1]: "<<(uint)hsi_pixel[1]<<std::endl;
    // std::cout<<"hsi_pixel[2]: "<<(uint)hsi_pixel[2]<<std::endl;
    



    return hsi_image;
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
  cv::namedWindow("Practice1", cv::WINDOW_AUTOSIZE);
  if (first){
    cv::createTrackbar("Option[0-6]", "Practice1", nullptr, 6, 0);
    // set Trackbar’s value
    cv::setTrackbarPos("Option[0-6]", "Practice1", 0);

    cv::createTrackbar("Filter value[0-100]", "Practice1", nullptr, 100, 0);
    // set Trackbar’s value
    cv::setTrackbarPos("Filter value[0-100]", "Practice1", 0);
    
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

  // get Trackbar’s value 
  int option = cv::getTrackbarPos("Option[0-6]", "Practice1");
  if (option == 0){
    cv::imshow("Practice1", out_image_rgb);
  }
  
  else if (option == 1){//show hsi image
    cv::Mat hsi_image = in_image_rgb.clone();
    // std::vector<cv::Mat> three_channels;
    // cv::split(hsi_image, three_channels);
    hsi_image = rgb_to_hsi(in_image_rgb);
    
    
    cv::imshow("Practice1", hsi_image);
  }
  else if (option == 2){
    cv::Mat hsi_image = in_image_rgb.clone();
    std::vector<cv::Mat> three_channels;
    cv::split(hsi_image, three_channels);
    hsi_image = rgb_to_hsi(in_image_rgb);
    
    cv::Mat HsvImage;
    cv::cvtColor(out_image_rgb, HsvImage, cv::COLOR_BGR2HSV);
    //cv::imshow("Practice1", HsvImage);
    
    //cv::Mat hsv_hsi_image(out_image_rgb.rows, out_image_rgb.cols, CV_32FC3);//imagen resultante de restar los canles hsv - hsi 

    std::vector<cv::Mat> hsv_channels;
    split(HsvImage, hsv_channels);
    
    std::vector<cv::Mat> hsi_channels;
    split(hsi_image, hsi_channels);

    
    cv::Mat result_Img;
    
    cv::Mat canal_A = cv::Mat::zeros(HsvImage.rows, HsvImage.cols, CV_8UC1);
    cv::Mat canal_B = cv::Mat::zeros(HsvImage.rows, HsvImage.cols, CV_8UC1);
    cv::Mat canal_C = cv::Mat::zeros(HsvImage.rows, HsvImage.cols, CV_8UC1);
    std::vector<cv::Mat> result_Img_vect;
    

    canal_A = hsv_channels[0] - hsi_channels[0];
    canal_B = hsv_channels[1] - hsi_channels[1];
    canal_C = hsv_channels[2] - hsi_channels[2];

    

    result_Img_vect.push_back(canal_A);
    result_Img_vect.push_back(canal_B);
    result_Img_vect.push_back(canal_C);
    
    cv::merge(result_Img_vect, result_Img);
    cv::imshow("Practice1", result_Img);



  }
  else if(option==3){
    cv::Mat I;
    cv::cvtColor(out_image_rgb, out_image_rgb, cv::COLOR_BGR2GRAY);

    cv::Mat complexImg = computeDFT(out_image_rgb);
    cv::Mat spectrum_original = spectrum(complexImg);

    cv::imshow("Practice1", spectrum_original);
  }
  else if(option==4){

    cv::Mat inverseTransform = option4(out_image_rgb);

    cv::imshow("Practice1", inverseTransform );

  }
  else if (option==5){

    cv::Mat inverseTransform = option5(out_image_rgb);

    cv::imshow("Practice1", inverseTransform );
  }
  else if(option==6){
    cv::Mat inverseTransform4 = option4(out_image_rgb);
    inverseTransform4.convertTo(inverseTransform4, CV_8UC1, 255.0);    
    
    cv::Mat um6(inverseTransform4.rows, inverseTransform4.cols, CV_8UC1);
    uint threshold_p4 = 153;

    for ( int i=0; i<inverseTransform4.rows; i++ ) {
      for ( int j=0; j<inverseTransform4.cols; j++ ) {        
        uint value = (uint)inverseTransform4.at<uchar>(i,j);
        if (value > threshold_p4) {
        um6.at<uchar>(i,j) = (uint)255;
        } else {
        um6.at<uchar>(i,j) = (uint)0;
        }
      }
    }    


    cv::Mat inverseTransform5 = option5(out_image_rgb);
    inverseTransform5.convertTo(inverseTransform5, CV_8UC1, 255.0);

    cv::Mat um4(inverseTransform5.rows, inverseTransform5.cols, CV_8UC1);
    uint threshold_p5 = 102;

    for ( int i=0; i<inverseTransform5.rows; i++ ) {
      for ( int j=0; j<inverseTransform5.cols; j++ ) {
        uint value = (uint)inverseTransform5.at<uchar>(i,j);
        if (value > threshold_p5)
        um4.at<uchar>(i,j) = (uint)255;
        else
        um4.at<uchar>(i,j) = (uint)0;
        }
    }


    int filterSize = cv::getTrackbarPos("Filter value[0-100]", "Practice1");
    if (filterSize == 0){
      filterSize = 1;
    }
    
    cv::cvtColor(out_image_rgb, out_image_rgb, cv::COLOR_BGR2GRAY);
    cv::Mat complexImg = computeDFT(out_image_rgb);
    cv::Mat spectrum_original = spectrum(complexImg);
    cv::Mat shift_complex = fftShift(complexImg);
    
    //shift_complex.convertTo(shift_complex, CV_8UC1, 255.0);

    cv::Mat im_white = cv::Mat::zeros(shift_complex.size(), shift_complex.type());
    cv::Mat im_black = cv::Mat::ones(shift_complex.size(), shift_complex.type());

    int cy = complexImg.rows / 2;

    cv::line(im_white, cv::Point(0,  cy), cv::Point(im_white.cols, cy), cv::Scalar(255,255,255),  filterSize);
    cv::line(im_black, cv::Point(0,  cy), cv::Point(im_black.cols, cy), cv::Scalar(0,0,0),  filterSize);

    cv::Mat output1, output2;

    cv::mulSpectrums(im_white, shift_complex, output1, 0);
    cv::mulSpectrums(im_black, shift_complex, output2, 0);

    cv::Mat img_princ;
    cv::bitwise_or(um4, um6, img_princ);
    cv::imshow("Practice1", img_princ);
  

    cv::Mat shift_complex1 = fftShift(output1);
    cv::Mat shift_complex2 = fftShift(output2);
    cv::Mat spectrum1 = spectrum(shift_complex1);
    cv::Mat spectrum2 = spectrum(shift_complex2);

    int key = cv::waitKey(100);;
    if (key == 100){
      if (pulse == 1){
        pulse = 0;
        key = 0;
        cv::destroyWindow("Umbral 0.6");
        cv::destroyWindow("Umbral 0.4");
        cv::destroyWindow("Output1");
        cv::destroyWindow("Output2");
      } 
      else if (pulse == 0){
        
        pulse = 1;
        key = 0;
      }
    }
    if (pulse == 1) {
      cv::imshow("Umbral 0.6", um6);
      cv::imshow("Umbral 0.4", um4);
      cv::imshow("Output1", spectrum1);
      cv::imshow("Output2", spectrum2);
    }
    else if(pulse == 0){
      
    }
  }
  
  else{
    cv::waitKey(3);
    return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
  }

  
  cv::waitKey(3);

  return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
}

  

} // namespace computer_vision

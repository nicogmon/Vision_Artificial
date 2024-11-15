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
using namespace cv;
using namespace std;
bool first = true;
string winname = "Practice2";
int n_contours = 0;
vector<Scalar> colors;


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

    return spectrum;
  }

  vector<Scalar> color_generator(long unsigned int n) {
    vector<Scalar> colors; // Vector to store the colors 
    while (colors.size() < n) {
        int num1 = rand() % 256; // Generate a random number between 0 and 255
        int num2 = rand() % 256;
        int num3 = rand() % 256;
        Scalar color(num1, num2, num3);
        colors.push_back(color);
    }
    return colors;
}


namespace computer_vision
{

CVGroup CVSubscriber::processing(
  const cv::Mat in_image_rgb,
  const cv::Mat in_image_depth,
  const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
const
{
  cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
  if (first){
    cv::createTrackbar("Option[0-4]", winname, nullptr, 4, 0); // create Trackbar
    cv::setTrackbarPos("Option[0-4]", winname, 0); // set Trackbar’s value

    cv::createTrackbar("Shrink min value[0-125]", winname, nullptr, 125, 0);
    cv::setTrackbarPos("Shrink min value[0-125]", winname, 0);

    cv::createTrackbar("Shrink max value[126-255]", winname, nullptr, 129, 0);
    cv::setTrackbarPos("Shrink max value[126-255]", winname, 0);

    cv::createTrackbar("Hough accumulator[0-255]", winname, nullptr, 255, 0);
    cv::setTrackbarPos("Hough accumulator[0-255]", winname, 0);

    cv::createTrackbar("Area[0-500]", winname, nullptr, 500, 0);
    cv::setTrackbarPos("Area[0-500]", winname, 0);    
    
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

  if (option != 1){
    cv::destroyWindow("Histogram");
  }

  if (option == 0){
    cv::imshow(winname, out_image_rgb);
  }
  else if (option == 1){
    // Parameters for the histogram
    int histSize = 256;
    float range[] = {0, 256};       // the upper boundary is exclusive
    const float * histRange = {range};
    bool uniform = true, accumulate = false;

    int min = cv::getTrackbarPos("Shrink min value[0-125]", winname);
    int max = (125 + cv::getTrackbarPos("Shrink max value[126-255]", winname));

    cv::Mat gray;
    cv::cvtColor(out_image_rgb, gray, cv::COLOR_BGR2GRAY);
    Mat gray_hist;
    calcHist(&gray, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate);

    // DFT
    cv::Mat complexImg = computeDFT(gray);
    cv::Mat spectrum_original = spectrum(complexImg);
    cv::Mat shift_complex = fftShift(complexImg);
    int cx = shift_complex.cols / 2;
    int cy = shift_complex.rows / 2;

    int filterSize = 70;

    Mat filter = Mat::zeros(shift_complex.size(), shift_complex.type());
    cv::circle(filter, cv::Point(cx, cy), filterSize, cv::Scalar(1), -1);

    mulSpectrums(filter, shift_complex, shift_complex, 0);
    cv::Mat rearrange = fftShift(shift_complex);

    cv::Mat inverseTransform(gray.size(), CV_8UC1);
    
    cv::idft(rearrange, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    
    cv::normalize(inverseTransform, inverseTransform, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    // Stretching the histogram
    double minVal, maxVal;
    cv::minMaxLoc(inverseTransform, &minVal, &maxVal);
    minVal = (uint)minVal;
    maxVal = (uint)maxVal;
    min = (uint)min;
    max = (uint)max;
  
    for (int i = 0; i < inverseTransform.rows; i++) {
      for (int j = 0; j < inverseTransform.cols; j++) {
         inverseTransform.at<uchar>(i, j) = (((max - min)/(maxVal - minVal)) * ((uint)inverseTransform.at<uchar>(i, j) - minVal)) + min;
      }
    }

    Mat contHist;
    calcHist(&inverseTransform, 1, 0, Mat(), contHist, 1, &histSize, &histRange, uniform, accumulate);
        
    Mat substract(inverseTransform.size(), inverseTransform.type());
    
    substract = inverseTransform - gray;

    Mat subshist;
    calcHist(&substract, 1, 0, Mat(), subshist, 1, &histSize, &histRange, uniform, accumulate);
    
    // Expand the histogram
    int MAX = 255;
    int MIN = 0;

    Mat expand = substract.clone();
    double minVal2, maxVal2;
    cv::minMaxLoc(expand, &minVal2, &maxVal2);
    minVal = (uint)minVal;
    maxVal = (uint)maxVal;
    
    for (int i = 0; i < expand.rows; i++) {
      for (int j = 0; j < expand.cols; j++) {
        expand.at<uchar>(i, j) = (((((uint)expand.at<uchar>(i, j)) - minVal2)/(maxVal - minVal)) * (MAX - MIN)) + MIN;
      }
    }

    Mat expand_hist;
    calcHist(&expand, 1, 0, Mat(), expand_hist, 1, &histSize, &histRange, uniform, accumulate);
    
    //Histogram equalization
    Mat eq_hist;
    equalizeHist(substract, eq_hist);
    
    Mat hist_eq;
    calcHist(&eq_hist, 1, 0, Mat(), hist_eq, 1, &histSize, &histRange, uniform, accumulate);    

    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0) );

    normalize(gray_hist, gray_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(contHist, contHist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(subshist, subshist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(expand_hist, expand_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(hist_eq, hist_eq, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    for (int i = 1; i < histSize; i++) {
      line(
        histImage, Point(bin_w * (i - 1), hist_h - cvRound(gray_hist.at<float>(i - 1)) ),
        Point(bin_w * (i), hist_h - cvRound(gray_hist.at<float>(i)) ),
        Scalar(255, 0, 0), 2, 8, 0);
      line(
        histImage, Point(bin_w * (i - 1), hist_h - cvRound(contHist.at<float>(i - 1)) ),
        Point(bin_w * (i), hist_h - cvRound(contHist.at<float>(i)) ),
        Scalar(0, 0, 255), 2, 8, 0);  
      line(
        histImage, Point(bin_w * (i - 1), hist_h - cvRound(subshist.at<float>(i - 1)) ),
        Point(bin_w * (i), hist_h - cvRound(subshist.at<float>(i)) ),
        Scalar(255, 255, 0), 2, 8, 0);
      line(
        histImage, Point(bin_w * (i - 1), hist_h - cvRound(expand_hist.at<float>(i - 1)) ),
        Point(bin_w * (i), hist_h - cvRound(expand_hist.at<float>(i)) ),
        Scalar(0, 255, 255), 2, 8, 0);
      line(
        histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist_eq.at<float>(i - 1)) ),
        Point(bin_w * (i), hist_h - cvRound(hist_eq.at<float>(i)) ),
        Scalar(0, 255, 0), 2, 8, 0);
    }
    
    double cont_gray = compareHist(gray_hist, contHist, HISTCMP_CORREL); 
    double sub_gray = compareHist(gray_hist, subshist, HISTCMP_CORREL);
    double expand_gray = compareHist(gray_hist, expand_hist, HISTCMP_CORREL);
    double eq_gray = compareHist(gray_hist, hist_eq, HISTCMP_CORREL);

    cv::putText(histImage, "Shrink ["+ to_string(min)+", " + to_string(max) + "]: " + to_string(cont_gray), Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
    cv::putText(histImage, "Substract: "+ to_string(sub_gray), Point(5, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
    cv::putText(histImage, "Stretch: " + to_string(expand_gray), Point(5, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
    cv::putText(histImage, "Eqhist: " + to_string(eq_gray), Point(5, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

    cv::imshow("Histogram", histImage);
    imshow(winname, eq_hist);
  
  }


  else if (option == 2){
    Mat dst, gray;

    cvtColor(out_image_rgb, gray, COLOR_BGR2GRAY);

    int accumulator = getTrackbarPos("Hough accumulator[0-255]", winname);

    // Edge detection
    Canny(gray, dst, 50, 150, 3);

    // Standard Hough Line Transform
    vector<Vec2f> lines;   // will hold the results of the detection (rho, theta)
    HoughLines(dst, lines, 1, CV_PI / 180, accumulator, 0, 0);   // runs the actual detection

    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++) {
      float rho = lines[i][0], theta = lines[i][1];
      Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a * rho, y0 = b * rho;
      pt1.x = cvRound(x0 + 1000 * (-b));
      pt1.y = cvRound(y0 + 1000 * ( a));
      pt2.x = cvRound(x0 - 1000 * (-b));
      pt2.y = cvRound(y0 - 1000 * ( a));
      line(out_image_rgb, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
    }

    // Show results
    imshow(winname, out_image_rgb);
    
  }
  else if (option == 3){
    Mat gray,mask, filteredImg, hsv;

    cvtColor(out_image_rgb, gray, COLOR_BGR2GRAY);
    cvtColor(out_image_rgb, hsv, cv::COLOR_BGR2HSV);

    Scalar lowerBrown = Scalar(10, 100, 20);
    Scalar upperBrown = Scalar(20, 255, 200);
  
    cv::inRange(hsv, lowerBrown, upperBrown, mask);
    
    bitwise_and(gray, gray, filteredImg, mask);
    
    vector<Vec3f> circles;
    HoughCircles(
      filteredImg, circles, HOUGH_GRADIENT, 0.1,
      filteredImg.rows / 10,             // change this value to detect circles with different distances to each other
      150, 35, 1, 150              // change the last two parameters (min_radius & max_radius) to detect larger circles
    );

    for (size_t i = 0; i < circles.size(); i++) {
      Vec3i c = circles[i];
      Point center = Point(c[0], c[1]);
      // circle center
      circle(out_image_rgb, center, 1, Scalar(0, 255, 0), 3, LINE_AA);
      // circle outline
      int radius = c[2];
      circle(out_image_rgb, center, radius, Scalar(0, 0, 255), 3, LINE_AA);
    }
    imshow(winname, out_image_rgb);
  }
  else if (option == 4){
    Mat dst,  cdstP, hsv, blue_mask, mask_marco, gray, mask_union, gauss;
    
    cvtColor(out_image_rgb, gray, COLOR_BGR2GRAY);
    cvtColor(out_image_rgb, hsv, COLOR_BGR2HSV);

    Scalar lowerMarco = Scalar(70, 0, 0);
    Scalar upperMarco = Scalar(360, 255, 100);
    inRange(hsv, lowerMarco, upperMarco, mask_marco);
    
    Scalar lowerBlue = Scalar(80, 20, 120);
    Scalar upperBlue = Scalar(100, 40, 220);
    cv::inRange(hsv, lowerBlue, upperBlue, blue_mask);
    
    bitwise_or(mask_marco, blue_mask, mask_union);

    // Contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask_union, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    if(n_contours == 0){
      n_contours = contours.size();
      
      colors = color_generator(200);
    }

    vector<Moments> mu(contours.size() );
    vector<Point2f> centroids(contours.size());

    for (size_t i = 0; i < contours.size(); i++) {
      if ((int) contourArea(contours[i], false) > getTrackbarPos("Area[0-500]", winname)) {
        drawContours(out_image_rgb, contours, i, colors[i], 2, LINE_8, hierarchy, 0);
        mu[i] = moments(contours[i]);
        centroids[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
        circle(out_image_rgb, centroids[i], 4, colors[i], -1, 8, 0);
        double pixels = contourArea(contours[i], false);
        putText(out_image_rgb, std::to_string(pixels), Point(centroids[i].x + 5, centroids[i].y), FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1);
      }
    }
    imshow(winname, out_image_rgb); 
  }
    
  cv::waitKey(3);

  return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
}

} // namespace computer_vision

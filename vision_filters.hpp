#pragma once // Prevents the compiler from including this file twice

#include <opencv4/opencv2/opencv.hpp>

// Declare the functions (No logic here, just the signatures)
cv::Mat draw_gray_img(const cv::Mat &img, int start, int end, int val);
cv::Mat draw_color_img(const cv::Mat &img, int start, int end, int b, int g, int r);
cv::Mat crop_image(const cv::Mat &img, int x, int y, int w, int h);
cv::Mat resize_image(const cv::Mat& img, int x = 244, int y = 244);
cv::Mat custom_grayscale(const cv::Mat &color_img);
cv::Mat custom_blur_filter(const cv::Mat &color_img);
cv::Mat pad_img_gray(const cv::Mat &input_img);
cv::Mat pad_img_color(const cv::Mat &input_img);
cv::Mat custom_sobel(const cv::Mat &gray_img);

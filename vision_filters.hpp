#pragma once // Prevents the compiler from including this file twice
#include <vector>
#include <opencv4/opencv2/opencv.hpp>

void draw_gray_img(const cv::Mat &img, cv::Mat &gray_img, int start, int end, int val);
void draw_color_img(const cv::Mat &img, cv::Mat &color_img, int start, int end, int b, int g, int r);
void crop_image(const cv::Mat &img, cv::Mat &cropped, int x, int y, int w, int h);
void resize_image(const cv::Mat& img, cv::Mat &color_img, int x = 244, int y = 244);
void custom_grayscale(const cv::Mat &color_img, cv::Mat &gray_out);
void custom_blur_filter(const cv::Mat &color_img, cv::Mat &blur_out);
void pad_img_gray(const cv::Mat &input_img, cv::Mat &padded_out);
void pad_img_color(const cv::Mat &input_img, cv::Mat &padded_col_out);
void custom_sobel(const cv::Mat &gray_img, cv::Mat &sobel_out);
void prepare_tensor(const cv::Mat &input, std::vector<float> &tensor_data);

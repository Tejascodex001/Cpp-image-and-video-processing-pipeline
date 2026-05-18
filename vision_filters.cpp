#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <vector>
#include "vision_filters.hpp"
#include <opencv2/core/types.hpp>

using namespace cv;
using namespace std;

// function for resizing the image
void resize_image(const Mat& img, Mat & resized, int x, int y){
    Size sized(x,y);
    resize(img, resized, sized);
}

// function for cropping the image
void crop_image(const Mat &img, Mat &cropped, int x, int y, int w, int h){
    Rect crp(x,y,w,h);
    img(crp).copyTo(cropped);
}

// function for drawing on color image
void draw_color_img(const Mat &img, Mat &color_img,int start, int end, int b, int g, int r){
    img.copyTo(color_img);
    for(int i = start; i < start+end; i++){
        Vec3b *row_ptr = color_img.ptr<Vec3b>(i);
        for(int j = start; j < start+end; j++){
            row_ptr[j] = Vec3b(b,g,r);
        }
    }
}

// function for drawing on gray image
void draw_gray_img(const Mat &img, Mat &gray_img, int start, int end, int val){
    img.copyTo(gray_img);
    for(int i = start; i < (start+end); i++){
        uchar *row_ptr = gray_img.ptr<uchar>(i);
        for(int j = start; j < (start+end); j++)
            row_ptr[j] = val;
    }
}

// function for custom grayscale image creation using bgr image
void custom_grayscale(const Mat &color_img, Mat &gray_out){
    gray_out.create(color_img.rows, color_img.cols, CV_8UC1);
    for(int i = 0; i < color_img.rows; i++){
        const Vec3b *color_ptr = color_img.ptr<Vec3b>(i);
        uchar *gray_ptr = gray_out.ptr<uchar>(i);
        for(int j = 0; j < color_img.cols; j++){
            uchar Blue = color_ptr[j][0], Green = color_ptr[j][1], Red = color_ptr[j][2];
            double Gray = (Blue * 0.114) + (Green * 0.587) + (Red * 0.299);
            gray_ptr[j] = (uchar)Gray;
        }
    }
}


void custom_blur_filter(const Mat &color_img, Mat &blur_out){
    blur_out.create(color_img.rows, color_img.cols, CV_8UC3);
    blur_out.setTo(Scalar(0,0,0));
    for(int i = 1; i < color_img.rows-1; i++){
        const Vec3b *prev_row = color_img.ptr<Vec3b>(i-1);
        const Vec3b *curr_row = color_img.ptr<Vec3b>(i);
        const Vec3b *next_row = color_img.ptr<Vec3b>(i+1);
        Vec3b *out_row = blur_out.ptr<Vec3b>(i);
        for(int j = 1; j < color_img.cols-1; j++){
            int sum_B = prev_row[j-1][0] + prev_row[j][0] + prev_row[j+1][0] +
                        curr_row[j-1][0] + curr_row[j][0] + curr_row[j+1][0] +
                        next_row[j-1][0] + next_row[j][0] + next_row[j+1][0];

            int sum_G = prev_row[j-1][1] + prev_row[j][1] + prev_row[j+1][1] +
                        curr_row[j-1][1] + curr_row[j][1] + curr_row[j+1][1] +
                        next_row[j-1][1] + next_row[j][1] + next_row[j+1][1]; 

            int sum_R = prev_row[j-1][2] + prev_row[j][2] + prev_row[j+1][2] +
                        curr_row[j-1][2] + curr_row[j][2] + curr_row[j+1][2] +
                        next_row[j-1][2] + next_row[j][2] + next_row[j+1][2];

            out_row[j] = Vec3b(sum_B/9, sum_G/9, sum_R/9);
        }
    }
}

// custom padding for grayimg
void pad_img_gray(const Mat &input_img, Mat &padded_out){
    int new_rows = input_img.rows +2, new_cols = input_img.cols + 2;
    padded_out.create(new_rows, new_cols, CV_8UC1);
    padded_out.setTo(0);
    for(int i = 0; i < input_img.rows; i++){
       const uchar *img_ptr = input_img.ptr<uchar>(i);
        uchar *pad_ptr = padded_out.ptr<uchar>(i+1);
        for(int j = 0; j < input_img.cols; j++){
            pad_ptr[j+1] = img_ptr[j];
        }
    }
}

// custom padding for color img
void pad_img_color(const Mat &input_img, Mat &padded_col_out){
    int new_rows = input_img.rows+2, new_cols = input_img.cols+2;
    padded_col_out.create(new_rows, new_cols, CV_8UC3);
    padded_col_out.setTo(Scalar(0,0,0));
    for(int i = 0; i < input_img.rows; i++){
        const Vec3b *img_ptr = input_img.ptr<Vec3b>(i);
        Vec3b *pad_ptr = padded_col_out.ptr<Vec3b>(i+1);
        for(int j = 0; j < input_img.cols; j++){
            pad_ptr[j+1] = img_ptr[j]; 
        }
    }
}

// custom sobel -> google sobel computer vision opencv or ask LLM
void custom_sobel(const Mat &gray_img, Mat &sobel_out){
    Mat padded;
    pad_img_gray(gray_img, padded);
    sobel_out.create(gray_img.rows, gray_img.cols, CV_8UC1);
    sobel_out.setTo(0);
    
    for(int i = 1; i < padded.rows-1; i++){
        const uchar *prev = padded.ptr<uchar>(i-1);
        const uchar *curr = padded.ptr<uchar>(i);
        const uchar *next = padded.ptr<uchar>(i+1);
        uchar *out_row = sobel_out.ptr<uchar>(i-1);
        
        for(int j = 1; j < padded.cols-1; j++){
            // gx kernel                                      structure of gx
            int Gx = (prev[j-1] * -1) + (prev[j+1] * 1) +   //  [-1, 0 ,1]
                     (curr[j-1] * -2) + (curr[j+1] * 2) +   //  [-2, 0, 2]
                     (next[j-1] * -1) + (next[j+1] * 1);    //  [-1, 0 ,1]
            

            //gy kernel                                                        structure of gy
                                                                                // [-1, -2, -1]
            int Gy = (prev[j-1] * -1) + (prev[j] * -2) + (prev[j+1] * -1) +     // [0, 0, 0] 
                     (next[j-1] * 1) + (next[j] * 2) + (next[j+1] *1);          // [1 ,2 ,1]
            
            int magn = abs(Gx) + abs(Gy);
            magn = min(255, magn);
            out_row[j-1] = (uchar)magn; 
        }
    }
}

// Converts OpenCV HWC (BGR) to AI NCHW (RGB) Float32
void prepare_tensor(const Mat &input, vector<float> &tensor_data) {
    int rows = input.rows, cols = input.cols;
    int total_pixels = rows*cols;

    // 1. Pre-allocate the exact float32 memory required
    // 3 channels * total pixels
    tensor_data.resize(3 * total_pixels);

    // 2. Setup memory pointers for the NCHW planar layout
    // tensor_data.data() returns a raw C-style pointer to the underlying array
    float* r_plane = tensor_data.data();
    float* g_plane = tensor_data.data() + total_pixels;
    float* b_plane = tensor_data.data() + (total_pixels * 2);

    // 3. Traverse the image and route the bytes to the correct planes
    for (int i = 0; i < rows; ++i) {
        const uchar* row_ptr = input.ptr<uchar>(i);
        
        for (int j = 0; j < cols; ++j) {
            uchar b = row_ptr[j * 3 + 0];
            uchar g = row_ptr[j * 3 + 1];
            uchar r = row_ptr[j * 3 + 2];

            // Calculate the flat 1D index for the current pixel
            int pixel_idx = (i * cols) + j;

            // Normalize (divide by 255.0f) and write to planar RGB layout
            r_plane[pixel_idx] = r / 255.0f;
            g_plane[pixel_idx] = g / 255.0f;
            b_plane[pixel_idx] = b / 255.0f;
        }
    }
}

void normalize_to_float32(const Mat &input, Mat &output){
    output.create(input.rows, input.cols, CV_32FC1);
    for(int i = 0; i < input.rows; i++){
        const uchar *in_ptr = input.ptr<uchar>(i);
        float* out_ptr = output.ptr<float>(i);
        for(int j = 0; j< input.cols; j++){
            out_ptr[j] = in_ptr[j]/255.0f;
        }
    }
}

void get_tensor_min_max(const Mat &float_32f, double &min_val, double &max_val){
    if(float_32f.empty()){
        min_val = 0.0f; max_val = 0.0f;
        return;
    }
    minMaxLoc(float_32f, &min_val, &max_val);

}

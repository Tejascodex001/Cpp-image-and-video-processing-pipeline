#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <sys/types.h>
#include <string>

using namespace std;
using namespace cv;

// function for resizing the image
Mat resize_image(Mat& img, int x = 224, int y = 224){
    Mat resized;
    Size sized(x,y);
    resize(img, resized, sized);
    return resized;
}

// function for cropping the image
Mat crop_image(Mat &img, int x, int y, int w, int h){
    Rect crp(x,y,w,h);
    Mat cropped = img(crp).clone();
    return cropped;
}

// function for drawing on color image
Mat draw_color_img(Mat &img, int start, int end, int b, int g, int r){
    Mat color_img = img.clone();
    for(int i = start; i < start+end; i++){
        Vec3b *row_ptr = color_img.ptr<Vec3b>(i);
        for(int j = start; j < start+end; j++){
            row_ptr[j] = Vec3b(b,g,r);
        }
    }
    return color_img;
}

// function for drawing on gray image
Mat draw_gray_img(Mat &img, int start, int end, int val){
    Mat gray_img = img.clone();
    for(int i = start; i < (start+end); i++){
        uchar *row_ptr = gray_img.ptr<uchar>(i);
        for(int j = start; j < (start+end); j++)
            row_ptr[j] = val;
    }
    return gray_img;
}

// function for custom grayscale image creation using bgr image
Mat custom_grayscale(const Mat &color_img){
    Mat gray_out(color_img.rows, color_img.cols, CV_8UC1);
    for(int i = 0; i < color_img.rows; i++){
        const Vec3b *color_ptr = color_img.ptr<Vec3b>(i);
        uchar *gray_ptr = gray_out.ptr<uchar>(i);
        for(int j = 0; j < color_img.cols; j++){
            uchar Blue = color_ptr[j][0], Green = color_ptr[j][1], Red = color_ptr[j][2];
            double Gray = (Blue * 0.114) + (Green * 0.587) + (Red * 0.299);
            gray_ptr[j] = (uchar)Gray;
        }
    }
    return gray_out;
}


Mat custom_blur_filter(const Mat &color_img){
    Mat blur_out = Mat::zeros(color_img.rows, color_img.cols, CV_8UC3);
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
    return blur_out;
}

// custom padding for grayimg
Mat pad_img_gray(const Mat &input_img){
    int new_rows = input_img.rows +2, new_cols = input_img.cols + 2;
    Mat padded_out = Mat::zeros(new_rows, new_cols, CV_8UC1);
    for(int i = 0; i < input_img.rows; i++){
       const uchar *img_ptr = input_img.ptr<uchar>(i);
        uchar *pad_ptr = padded_out.ptr<uchar>(i+1);
        for(int j = 0; j < input_img.cols; j++){
            pad_ptr[j+1] = img_ptr[j];
        }
    }
    return padded_out;
}

// custom padding for color img
Mat pad_img_color(const Mat &input_img){
    int new_rows = input_img.rows+2, new_cols = input_img.cols+2;
    Mat padded_col_out = Mat::zeros(new_rows, new_cols, CV_8UC3);
    for(int i = 0; i < input_img.rows; i++){
        const Vec3b *img_ptr = input_img.ptr<Vec3b>(i);
        Vec3b *pad_ptr = padded_col_out.ptr<Vec3b>(i+1);
        for(int j = 0; j < input_img.cols; j++){
            pad_ptr[j+1] = img_ptr[j]; 
        }
    }
    return padded_col_out;
}

// custom sobel -> google sobel computer vision opencv or ask LLM
Mat custom_sobel(const Mat &gray_img){
    Mat padded = pad_img_gray(gray_img);
    Mat sobel_out = Mat::zeros(gray_img.rows, gray_img.cols, CV_8UC1);
    
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
    return sobel_out;
}


int main(int argc, char** argv){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
        return 1;
    }
    string img_path = argv[1];
    int start = 50, width = 50;

    
    Mat img = imread(img_path, IMREAD_COLOR);
     if(img.empty()){
        cout << "Couldn't read image" << img_path << endl;
        return 1;
    }

    cout << "number of channels: " << img.channels() << endl;

    Vec3b intensity = img.at<Vec3b>(50,50);
    uchar blue = intensity.val[0];
    uchar green = intensity.val[1];
    uchar red = intensity.val[2];

    cout << "blue: " << int(blue) << " green: " << int(green) << " red: " << int(red) << endl;
    
    // call of drawing on color image
    Mat res_clr = draw_color_img(img, start, width, 0, 0, 255);
    imwrite("draw_clr.jpg", res_clr);

    Mat grayimg;
    cvtColor(img, grayimg, COLOR_BGR2GRAY);   
    
    cout << "Image width: " << grayimg.cols << endl;
    cout << "Image height: " << grayimg.rows << endl;


    int pixel_val = grayimg.at<uchar>(50,50);
    cout << "pixel value at (50,50) before changing: " << pixel_val << endl;
    grayimg.at<uchar>(50,50) = 255;
    int pixval_ch = grayimg.at<uchar>(50,50);
    cout << "pixel value at (50,50) after changing: " << pixval_ch << endl;
    
    // call of cropping the image
    Mat res_crop = crop_image(grayimg, 10, 150, 50, 100);
    imwrite("res_crop.jpg", res_crop);

    // call of resize of image
    Mat res_resize = resize_image(grayimg); // optional parameters of x and y (size of image) can be passed. ex: (img, 244, 244);
    imwrite("res_resize.jpg", res_resize);                                            
                           
    // call of drawing on gray image
    Mat res_gray = draw_gray_img(grayimg, start, width, 255);
    imwrite("res_gray.jpg", res_gray);

    //custom grayscale for an image
    Mat cpimg = img.clone();
    Mat res_cgray = custom_grayscale(cpimg);
    imwrite("res_cgray.jpg", res_cgray);

    // custom blur filter 
    Mat res_cblur = custom_blur_filter(img);
    imwrite("res_cblur.jpg", res_cblur);

    // padding gray images
    Mat res_padgray = pad_img_gray(grayimg);
    imwrite("padded_img_gray.jpeg", res_padgray);
    
    // padding color images
    Mat res_padclr = pad_img_color(img);
    imwrite("padded_img_clr.jpeg", res_padclr);

    // custom sobel
    Mat res_csobel = custom_sobel(grayimg);
    imwrite("res_csobel.jpg", res_csobel);

    return 0;
}

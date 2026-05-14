#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <sys/types.h>
#include <string>
#include "vision_filters.hpp"

using namespace std;
using namespace cv;



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
    Mat res_clr;
    draw_color_img(img,res_clr, start, width, 0, 0, 255);
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
    Mat res_crop;
    crop_image(grayimg, res_crop, 10, 150, 50, 100);
    imwrite("res_crop.jpg", res_crop);

    // call of resize of image
    Mat res_resize ;
    resize_image(grayimg, res_resize); // optional parameters of x and y (size of image) can be passed. ex: (img, 244, 244);
    imwrite("res_resize.jpg", res_resize);                                            
                           
    // call of drawing on gray image
    Mat res_gray;
    draw_gray_img(grayimg, res_gray, start, width, 255);
    imwrite("res_gray.jpg", res_gray);

    return 0;
}

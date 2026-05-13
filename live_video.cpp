#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/videoio.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <sys/types.h>
#include "vision_filters.hpp"

using namespace std;
using namespace cv;

int main(){
    VideoCapture cap(0);
    if(!cap.isOpened()){
        cout << "Error: cannot open camera" << endl;
        return -1;
    }

    while(true){
        Mat Frame;
        cap >> Frame;
        if(Frame.empty()){
            cout << "Frame has dropped" << endl;
            break;
        }
        Frame = resize_image(Frame, 640, 360);
        Mat gray = custom_grayscale(Frame); // grayscale
        Mat blurred = custom_blur_filter(Frame); // applying blur
        Mat edge = custom_sobel(gray); // applying sobel edge detection

        Mat edges_bgr, gray_bgr;
        cvtColor(gray, gray_bgr, COLOR_GRAY2BGR);
        cvtColor(edge, edges_bgr, COLOR_GRAY2BGR);

        Mat combined_view_top;
        hconcat(Frame, gray_bgr, combined_view_top);

        Mat combined_view_bot;
        hconcat(blurred, edges_bgr, combined_view_bot);

        Mat combined_view;
        vconcat(combined_view_top, combined_view_bot, combined_view);

        imshow("Live: original vs edge/sobel", combined_view);
        if(waitKey(30) >= 0) break;
    }
}

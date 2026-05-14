#include <iostream>
#include <chrono>
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
    Mat Frame, small_frame;
    Mat gray, blurred, edge, edges_bgr, gray_bgr;
    Mat combined_view_top, combined_view_bot, combined_view;
    while(true){
        cap >> Frame;
        if(Frame.empty()){
            cout << "Frame has dropped" << endl;
            break;
        }

        auto start = std::chrono::high_resolution_clock::now(); // start the time

        resize_image(Frame, small_frame, 640, 360);
        custom_grayscale(small_frame, gray); // grayscale
        custom_blur_filter(small_frame, blurred); // applying blur
        custom_sobel(gray, edge); // applying sobel edge detection

        Mat edges_bgr, gray_bgr;
        cvtColor(gray, gray_bgr, COLOR_GRAY2BGR);
        cvtColor(edge, edges_bgr, COLOR_GRAY2BGR);

        hconcat(small_frame, gray_bgr, combined_view_top);
        hconcat(blurred, edges_bgr, combined_view_bot);
        vconcat(combined_view_top, combined_view_bot, combined_view);

        auto end = std::chrono::high_resolution_clock::now(); // stop the clock
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count(); // calculate the time elapsed
        cout << "Frame time: " << elapsed_ms << " ms" << endl;

        imshow("Live: original vs edge/sobel", combined_view);

        if(waitKey(1) >= 0) break;
    }
}

#include <iostream>
#include <chrono>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/videoio.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <ratio>
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
    std::vector<float> ai_tensor;

    // --- ROLLING ACCUMULATORS ---
    int frame_count = 0;
    double total_loop_ms = 0.0;
    double total_gray_ms = 0.0;
    double total_blur_ms = 0.0;
    double total_sobel_ms = 0.0;
    double total_tensor_ms = 0.0;
    
    Mat norm_float;
    double current_min = 0.0;
    double current_max = 0.0;
    bool enable_norm = false; // toggle via 'n'

    auto wall_clock_start = std::chrono::high_resolution_clock::now();

    // --- HUD DISPLAY STATE ---
    double disp_fps = 0.0;
    double disp_gray = 0.0;
    double disp_blur = 0.0;
    double disp_sobel = 0.0;
    double disp_tensor = 0.0;

    bool enable_gray = true;
    bool enable_blur = true;
    bool enable_sobel = true;
    bool enable_tensor = true;

    while(true){
        cap >> Frame;
        if(Frame.empty()){
            cout << "Frame has dropped" << endl;
            break;
        }

        auto start = std::chrono::high_resolution_clock::now(); // start the time

        resize_image(Frame, small_frame, 640, 360);
        
        auto t0 = std::chrono::high_resolution_clock::now();

        if(enable_gray || enable_sobel) custom_grayscale(small_frame, gray); // grayscale
        else {
            gray.create(small_frame.rows, small_frame.cols, CV_8UC1);
            gray.setTo(0);
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        if(enable_blur) custom_blur_filter(small_frame, blurred);// applying blur
        else {
            blurred.create(small_frame.rows, small_frame.cols, CV_8UC3);
            blurred.setTo(Scalar(0,0,0));                               
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        if(enable_sobel) custom_sobel(gray, edge); // applying sobel edge detection
        else {
            edge.create(small_frame.rows, small_frame.cols, CV_8UC1);
            edge.setTo(0);                                           
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        
        if(enable_tensor) prepare_tensor(small_frame, ai_tensor);

        auto t4 = std::chrono::high_resolution_clock::now();

        if(enable_norm){
            normalize_to_float32(gray, norm_float);
            get_tensor_min_max(norm_float, current_min, current_max);
        }
        else{
            norm_float.create(small_frame.rows, small_frame.cols, CV_32FC1);
            norm_float.setTo(0.0f);
            current_min = 0.0; current_max = 0.0;
        }

        if(enable_gray) cvtColor(gray, gray_bgr, COLOR_GRAY2BGR);
        else {
            gray_bgr.create(small_frame.rows, small_frame.cols, CV_8UC3);
            gray_bgr.setTo(Scalar(0,0,0));
        }
        
        if(enable_sobel) cvtColor(edge, edges_bgr, COLOR_GRAY2BGR);
        else{
            edges_bgr.create(small_frame.rows, small_frame.cols, CV_8UC3);
            edges_bgr.setTo(Scalar(0,0,0));
        }

        // Assembly the different windows into single dashboard
        hconcat(small_frame, gray_bgr, combined_view_top);
        hconcat(blurred, edges_bgr, combined_view_bot);
        vconcat(combined_view_top, combined_view_bot, combined_view);

        auto end = std::chrono::high_resolution_clock::now(); // stop the clock
        
        // Calculate deltas
        double loop_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gray_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double blur_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double sobel_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double tensor_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // Accumulate
        total_loop_ms += loop_ms;
        total_gray_ms += gray_ms;
        total_blur_ms += blur_ms;
        total_sobel_ms += sobel_ms;
        total_tensor_ms += tensor_ms;
        frame_count++;
        
        auto now = std::chrono::high_resolution_clock::now();
        double real_elapsed_ms = std::chrono::duration<double, std::milli>(now - wall_clock_start).count();
        
        if(real_elapsed_ms > 1000.0){
            disp_fps = frame_count; // if 1000.0 ms has passed, frame_count is fps 
            disp_gray = total_gray_ms / frame_count;
            disp_blur = total_blur_ms / frame_count;
            disp_sobel = total_sobel_ms / frame_count;
            disp_tensor = total_tensor_ms / frame_count;

            std::cout << "[TELEMETRY] FPS: " << disp_fps 
                << " | Gray: " << disp_gray << "ms"
                << " | Blur: " << disp_blur << "ms"
                << " | Sobel: " << disp_sobel << "ms"
                << " | Tensor: " << disp_tensor << "ms\n";

            // Reset accumulators for the next second
            frame_count = 0;
            total_gray_ms = 0.0;
            total_blur_ms = 0.0;
            total_sobel_ms = 0.0;
            total_tensor_ms = 0.0;

            wall_clock_start = now;

        } 

        char text_buffer[256];

        sprintf(text_buffer, "FPS: %.0f", disp_fps);
        putText(combined_view, text_buffer, Point(10,30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        
        sprintf(text_buffer, "Gray: %.2f ms", disp_gray);
        putText(combined_view, text_buffer, Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

        sprintf(text_buffer, "Blur: %.2f ms", disp_blur);
        putText(combined_view, text_buffer, Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

        sprintf(text_buffer, "Sobel: %.2f ms", disp_sobel);
        putText(combined_view, text_buffer, Point(10, 130), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        
        sprintf(text_buffer, "Tensor: %.2f ms", disp_tensor);
        putText(combined_view, text_buffer, Point(10, 170), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);

        sprintf(text_buffer, "Norm [Min: %.2f | Max: %.2f]", current_min, current_max);
        putText(combined_view, text_buffer, Point(10, 200), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        

        imshow("Live: Dashboard", combined_view);

        char key = (char)waitKey(1);
        if(key == 27) break;
        if(key == 'g') enable_gray = !enable_gray;
        if(key == 'b') enable_blur = !enable_blur;
        if(key == 's') enable_sobel = !enable_sobel;
        if(key == 't') enable_tensor = !enable_tensor;
        if(key == 'n') enable_norm  =!enable_norm;



    }
}

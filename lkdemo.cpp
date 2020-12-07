#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctype.h>

using namespace cv;
using namespace std;
using namespace chrono;

int main( int argc, char** argv )
{
    VideoCapture cap;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 100;
    bool needToInit = true;

    cv::CommandLineParser parser(argc, argv, "{@input||}");
    string input = parser.get<string>("@input");
    if( input.empty() )
        cap.open(0, CAP_V4L2);
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(CAP_PROP_FPS, 30);

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    Mat gray, prevGray, image, frame;
    vector<Point2f> points[2];
    auto t1 = high_resolution_clock::now(), t2 = high_resolution_clock::now();
    int resetTime = 0;
    for(;;)
    {
        cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);

        if( needToInit || resetTime > 10000 )
        {
            // automatic initialization
            // OpenCV 3.2:
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
            // OpenCV 4.2: extra '3'
            // goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            cout << setprecision(2) << fixed;
            needToInit = false;
            resetTime = 0;
        }
        else if( !points[0].empty() )
        {
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            float flowX = 0, flowY = 0;
            size_t i, k;
            for( i = k = 0; i < points[1].size(); i++ )
            {
                if( !status[i] )
                    continue;

                // add to flow
                flowX += points[1][i].x - points[0][i].x;
                flowY += points[1][i].y - points[0][i].y;

                points[1][k++] = points[1][i];
            }
            points[1].resize(k);

            // fps
            t2 = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
            resetTime += duration;
            t1 = t2;
            // hacky way to overwrite line?
            cout << "flow x: " << flowX / k << "\t flow y: " << flowY / k << "\t loop time: " << duration << "\r" << flush;
            cout << "                                                                                           " << "\r";
            flowX = flowY = 0;
        }

        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
    }

    return 0;
}

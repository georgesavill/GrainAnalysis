#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/// Global variables for Threshold
int threshold_value = 235;
int threshold_type = 0;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
// Global variables for countours
int thresh = 8;
int max_thresh = 255;
RNG rng(12345);

Mat cal, cal_gray, src, src_gray, dst;
const char* window_name = "Threshold";
const char* window_input_name = "Input";
const char* window_output_name = "Output";
const char* window_calibration = "Calibration";

double conversion_factor;


/// Function headers
void Calibration();
void ProcessImage();

int main(int argc, char** argv)
{
	cal = imread("..//data/calibration.jpg", IMREAD_COLOR); // Load calibration image
	if (cal.empty()) return -1; // If calibration image cant be found, return -1
	src = imread("../data/grain.jpg", IMREAD_COLOR); // Load input image
	if (src.empty()) return -1; // If input image cant be found, return -1

	cvtColor(cal, cal_gray, COLOR_BGR2GRAY); // Convert the calibration image to Gray
	cvtColor(src, src_gray, COLOR_BGR2GRAY); // Convert the input image to Gray

	namedWindow(window_input_name, WINDOW_AUTOSIZE); // Create a window to display input image
	namedWindow(window_name, WINDOW_AUTOSIZE); // Create a window to display threshold input
	namedWindow(window_output_name, WINDOW_AUTOSIZE); // Create a window to display contour output
	namedWindow(window_calibration, WINDOW_AUTOSIZE); // Create a window to display calibration

	Calibration();
	ProcessImage();

	for (;;)
	{
		char c = (char)waitKey(20);
		if (c == 27) // ESC
		{
			break;
		}
	}

}

void Calibration()
{
	Mat erodeElement, dilateElement, calDrawing;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	double calibrationArea = 0;

	/// Blur input image
	blur(cal_gray, cal_gray, Size(4, 4));

	/// Threshold image to create binary image
	threshold(cal_gray, cal_gray, threshold_value, max_BINARY_value, CV_THRESH_BINARY | CV_THRESH_OTSU);

	/// Erode and dilate image to reduce noise
	erodeElement = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(2, 2));
	dilateElement = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(2, 2));
	dilate(cal_gray, cal_gray, dilateElement);
	erode(cal_gray, cal_gray, erodeElement);

	/// Detect edges using canny
	Canny(cal_gray, cal_gray, thresh, thresh * 2, 3);

	/// Find contours
	findContours(cal_gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	calDrawing = Mat::zeros(cal_gray.size(), CV_8UC3);
	for (size_t i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(calDrawing, contours, (int)i, color, 1, 8, hierarchy, 0, Point());
	}

	/// Measure area of each contour
	for (int i = 0; i < contours.size(); i++)
	{
		calibrationArea = calibrationArea + contourArea(contours[i]);
	}

	/// Print text to output image
	putText(calDrawing, "Number = " + to_string(contours.size()), cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 200), 1, CV_AA);
	putText(calDrawing, "Area = " + to_string(calibrationArea), cvPoint(30, 70), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 200), 1, CV_AA);

	imshow(window_calibration, calDrawing);
	conversion_factor = calibrationArea / 100;

}

void ProcessImage()
{
	Mat canny_output, erodeElement, dilateElement;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	double totalArea = 0;

	/// Blur input image
	blur(src_gray, src_gray, Size(4, 4));
	imshow(window_input_name, src_gray);

	/// Threshold image to create binary image
	threshold(src_gray, dst, threshold_value, max_BINARY_value, CV_THRESH_BINARY | CV_THRESH_OTSU);

	/// Erode and dilate image to reduce noise
	erodeElement = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(2, 2));
	dilateElement = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(2, 2));
	dilate(dst, dst, dilateElement);
	erode(dst, dst, erodeElement);
	imshow(window_name, dst);

	/// Detect edges using canny
	Canny(dst, canny_output, thresh, thresh * 2, 3);

	/// Find contours
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (size_t i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, (int)i, color, 3, 8, hierarchy, 0, Point());
	}

	/// Measure area of each contour
	for (int i = 0; i < contours.size(); i++)
	{
		totalArea = totalArea + contourArea(contours[i]);
	}

	/// Print text to output image
	putText(drawing, "Number of grain = " + to_string(contours.size()), cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 200), 1, CV_AA);
	putText(drawing, "Total area of grain = " + to_string(totalArea / conversion_factor) + "mm2", cvPoint(30, 70), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 200), 1, CV_AA);
	putText(drawing, "Mean area of grain = " + to_string((totalArea/ conversion_factor) / contours.size()) + "mm2", cvPoint(30, 110), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 200), 1, CV_AA);

	imshow(window_output_name, drawing);

}
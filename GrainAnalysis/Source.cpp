#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/// Mat variables
Mat cal, cal_gray, src, src_gray, dst;
/// Variables for Threshold
int threshold_value = 235;
int const max_BINARY_value = 255;
/// Variables for erode and dilate
int const erode_dilate_factor = 2;
/// Variables for countours
int thresh = 8;
int max_thresh = 255;
RNG rng(12345);
/// Window names
const char* window_name = "Threshold";
const char* window_input_name = "Input";
const char* window_output_name = "Output";
const char* window_calibration = "Calibration";
/// Conversion from pixel to mm2 variable
double conversion_factor;

/// Function headers
void Calibration();
void ProcessImage();

int main(int argc, char** argv)
{
	/// Load calibration and input images
	cal = imread("..//data/calibration.jpg", IMREAD_COLOR);
	src = imread("../data/grain.jpg", IMREAD_COLOR);
	/// Check calibration and input images were loaded successfully
	if (cal.empty()) return -1;
	if (src.empty()) return -1;
	/// Convert calibration and input images to grayscale
	cvtColor(cal, cal_gray, COLOR_BGR2GRAY);
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	/// Create windows
	namedWindow(window_input_name, WINDOW_AUTOSIZE);
	namedWindow(window_name, WINDOW_AUTOSIZE);
	namedWindow(window_output_name, WINDOW_AUTOSIZE);
	namedWindow(window_calibration, WINDOW_AUTOSIZE);

	Calibration();
	ProcessImage();

	/// ESC to exit program
	while(true)
	{
		char c = (char)waitKey(20);
		if (c == 27) break;
	}
}

void Calibration()
{
	Mat erode_element, dilate_element, cal_drawing;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	double calibration_area = 0;

	/// Blur input image
	blur(cal_gray, cal_gray, Size(4, 4));
	/// Threshold image to create binary image
	threshold(cal_gray, cal_gray, threshold_value, max_BINARY_value, CV_THRESH_BINARY | CV_THRESH_OTSU);
	/// Erode and dilate image to reduce noise
	erode_element = getStructuringElement(MORPH_ELLIPSE, Size(2 * erode_dilate_factor + 1, 2 * erode_dilate_factor + 1), Point(erode_dilate_factor, erode_dilate_factor));
	dilate_element = getStructuringElement(MORPH_ELLIPSE, Size(2 * erode_dilate_factor + 1, 2 * erode_dilate_factor + 1), Point(erode_dilate_factor, erode_dilate_factor));
	dilate(cal_gray, cal_gray, dilate_element);
	erode(cal_gray, cal_gray, erode_element);
	/// Detect edges using canny
	Canny(cal_gray, cal_gray, thresh, thresh * 2, 3);
	/// Find contours
	findContours(cal_gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	/// Draw contours
	cal_drawing = Mat::zeros(cal_gray.size(), CV_8UC3);
	for (size_t i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(cal_drawing, contours, (int)i, color, 1, 8, hierarchy, 0, Point());
	}
	/// Measure area of calibration contour(s)
	for (int i = 0; i < contours.size(); i++)
	{
		calibration_area = calibration_area + contourArea(contours[i]);
	}
	/// Print text to output image
	putText(cal_drawing, "Number = " + to_string(contours.size()), cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 200), 1, CV_AA);
	putText(cal_drawing, "Area = " + to_string(calibration_area), cvPoint(30, 70), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 200), 1, CV_AA);
	/// Display result of calibration
	imshow(window_calibration, cal_drawing);
	/// Calculate conversion factor (contour drawn around a 1cm2 square, hence 100mm2)
	conversion_factor = calibration_area / 100;
}

void ProcessImage()
{
	Mat canny_output, erode_element, dilate_element;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	double total_area = 0;

	/// Blur input image
	blur(src_gray, src_gray, Size(4, 4));
	imshow(window_input_name, src_gray);
	/// Threshold image to create binary image
	threshold(src_gray, dst, threshold_value, max_BINARY_value, CV_THRESH_BINARY | CV_THRESH_OTSU);
	/// Erode and dilate image to reduce noise
	erode_element = getStructuringElement(MORPH_ELLIPSE, Size(2 * erode_dilate_factor + 1, 2 * erode_dilate_factor + 1), Point(erode_dilate_factor, erode_dilate_factor));
	dilate_element = getStructuringElement(MORPH_ELLIPSE, Size(2 * erode_dilate_factor + 1, 2 * erode_dilate_factor + 1), Point(erode_dilate_factor, erode_dilate_factor));
	dilate(dst, dst, dilate_element);
	erode(dst, dst, erode_element);
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
		total_area = total_area + contourArea(contours[i]);
	}
	/// Print text to output image
	putText(drawing, "Number of grain = " + to_string(contours.size()), cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 200), 1, CV_AA);
	putText(drawing, "Total area of grain = " + to_string(total_area / conversion_factor) + "mm2", cvPoint(30, 70), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 200), 1, CV_AA);
	putText(drawing, "Mean area of grain = " + to_string((total_area/ conversion_factor) / contours.size()) + "mm2", cvPoint(30, 110), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 200), 1, CV_AA);
	/// Display results output image
	imshow(window_output_name, drawing);
}
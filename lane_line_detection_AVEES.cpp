#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

struct BinaryParam
{
	int thresh;
	int l_thresh;
	int b_thresh;
	int max_val;
};

Mat unwarp(Mat frame, Point2f src[], Point2f dst[]) {
	Mat M = getPerspectiveTransform(src, dst);
	Mat warped(frame.rows, frame.cols, frame.type());
	warpPerspective(frame, warped, M, warped.size(), INTER_LINEAR, BORDER_CONSTANT);
	return warped;
}  

Mat pipeline(Mat frame, BinaryParam param) {
	//Mat gray_frame;
	//Mat sobelx;
	//Mat sobely;
	
	Mat hls_frame;
	Mat gaus_hls_frame;
	Mat bin_hls_frame;
	
	//Mat lab_frame;

	//cvtColor(frame, gray_frame, CV_BGR2GRAY);
	//Sobel(gray_frame, sobelx, CV_64F, 1, 0); 
	//Sobel(gray_frame, sobely, CV_64F, 0, 1); 
	
	cvtColor(frame, hls_frame, CV_BGR2HLS);
	vector<Mat> hls_images(3);
	split(hls_frame, hls_images);
	
	//cvtColor(frame, lab_frame, CV_BGR2Lab);
	//vector<Mat> lab_images(3);
	//split(lab_frame, lab_images);

	GaussianBlur(hls_images[1], gaus_hls_frame, Size(0, 0), 1);
	
	threshold(gaus_hls_frame, bin_hls_frame, param.l_thresh, param.max_val, THRESH_BINARY);
	
	return bin_hls_frame;
}

double gaussian(double x, double mu, double sig)
{
	return exp((-1)*pow(x - mu, 2.0) / (2 * pow(sig, 2.0)));
}


Mat getHistLane(Mat& frame_bev, int& last_line_point) {
	int hist[frame_bev.cols] = {0, };
	int left_point = 0;
	int center_point = 640;
	
	for(int j=0;j<frame_bev.rows;j++) {
		for(int i=0;i<frame_bev.cols;i++) {
			if((int)frame_bev.at<uchar>(j, i) == 255 && abs(center_point - i) < 400) hist[i]++;
		}
	}

	// make and apply gaussian filter
	if(last_line_point != 0) {
		// parameter
		int distrib_width = 400;
		double sigma = distrib_width / 12;

		// make gaussian distrib
		double weight_distrib[frame_bev.cols] = {0, };
		int start_idx = last_line_point - distrib_width / 2;
		for(int i = start_idx; i < start_idx + distrib_width; i++) {
			weight_distrib[i] = gaussian(i, last_line_point, sigma);
		}

		// apply gaussian distrib
		for(int i = 0; i < frame_bev.cols; i++) {
			hist[i] = hist[i] * weight_distrib[i];
		}
	}
	
	int histMax = 0;
	
	for(int i=0;i<frame_bev.cols;i++) {
		if(hist[i] > histMax) {
			histMax = hist[i];
			left_point = i;
		}
	}

	Mat imgHist(100, frame_bev.cols, CV_8U, Scalar(255));
	
        for(int i=0;i<frame_bev.cols;i++) {
              line(imgHist, Point(i, 100), Point(i, 100 - cvRound(hist[i] * 100 / (histMax + 0.000001))), Scalar(0));
        }
	
	if(abs(center_point - left_point) < 400) last_line_point = left_point;
	//else last_line_point = 0;
	
	return imgHist;
}

Mat mergeFrame(Mat& frame1, Mat& frame2) {
	Mat merge_frame(frame1.rows, frame1.cols, frame1.type(), Scalar(0));
	for(int j=0;j<frame1.rows;j++) {
		for(int i=0;i<frame1.cols;i++) {
			if(frame1.at<uchar>(j,i) == 255 && frame2.at<uchar>(j,i) == 255) merge_frame.at<uchar>(j,i) = 255;
			else merge_frame.at<uchar>(j,i) = 0;
		}
	}
	return merge_frame;
}

void slidingWindow(Mat& frame, Mat& frame_line, int& line_point, vector<Point2f>& points) {
	Mat frame_window = frame_line.clone();
	int center_point = line_point;
	int prev_center_point = line_point;
	int hist[128] = {0, };
	int histMax;
	int offset;
	int first_index = 0;
	int last_index = 0;
	
	for(int k=1;k<11;k++) {
		first_index = 0;
		last_index = 0;
		offset = center_point - 64;
		for(int j=frame.rows - 72 * k;j<=frame.rows - 72 * (k - 1);j++) {
			for(int i=center_point - 64;i<=center_point + 64;i++) {
				if(frame_line.at<uchar>(j, i) == 255) {
					hist[i - offset] = hist[i - offset] + 1;
				}
			}
		}
	
		
	
		for(int q=0;q<128;q++) {
			if(hist[q] != 0) {
				if(hist[q] > 20) {
					last_index = q;
					if(first_index == 0) first_index = q;
				}
			}
			if(hist[q] > histMax) {
				histMax = hist[q];
				center_point = q + offset;
			}
		}

		int gap = last_index - first_index;
		
		if(gap > 50) continue;	

		if(center_point >= 0 && histMax > 50 && abs(center_point - prev_center_point) < 200 ) {
			points.push_back(Point(720 - ((frame.rows - 72 * k) + (frame.rows - 72 * (k - 1))) / 2, 1280 - center_point));
			rectangle(frame, Point(center_point - 64, frame.rows - k * 72), Point(center_point + 64, frame.rows - 72 * (k - 1)), Scalar(255), 2);
		}
		else {
			center_point = prev_center_point;
		}
		histMax = 0;
		
		for(int r=0;r<128;r++) hist[r] = 0;
	}
}

Mat polyfit(vector<Point2f>& points) {
	Mat coef(3, 1, CV_32F);
	int i,j,k;
	int N = points.size();
	int n = 2;
	float x[N], y[N];
	for(int q=0;q<N;q++) {
		x[q] = points[q].x;
		y[q] = points[q].y;
	}
	float X[2*n+1];                        
    	for (i=0;i<2*n+1;i++)
    	{
        	X[i]=0;
        	for (j=0;j<N;j++)
            		X[i]=X[i]+pow(x[j],i);        
    	}
    	float B[n+1][n+2],a[n+1];            
    	for (i=0;i<=n;i++)
        	for (j=0;j<=n;j++)
            		B[i][j]=X[i+j];           
    	float Y[n+1];                    
    	for (i=0;i<n+1;i++)
    	{    
        	Y[i]=0;
        	for (j=0;j<N;j++)
        	Y[i]=Y[i]+pow(x[j],i)*y[j];        
    	}
    	for (i=0;i<=n;i++)
        	B[i][n+1]=Y[i];                
    	n=n+1;               
    	for (i=0;i<n;i++)                    
        	for (k=i+1;k<n;k++)
            		if (B[i][i]<B[k][i])
                		for (j=0;j<=n;j++)
                		{
                    			float temp=B[i][j];
                    			B[i][j]=B[k][j];
                   			B[k][j]=temp;
               			 }	
    
    	for (i=0;i<n-1;i++)           
        	for (k=i+1;k<n;k++)
           	{
               		 float t=B[k][i]/B[i][i];
              		 for (j=0;j<=n;j++)
                   		 B[k][j]=B[k][j]-t*B[i][j];    
            	}
    	for (i=n-1;i>=0;i--)               
    	{                        
        	a[i]=B[i][n];                
        	for (j=0;j<n;j++)
            		if (j!=i)
                		a[i]=a[i]-B[i][j]*a[j];
        	a[i]=a[i]/B[i][i];
		coef.at<float>(i, 0) = a[i];
    	}
	return coef;
}

float calcCurveRadius(vector<Point2f>& points) {
	float xm_per_pix = 1.000 / 640;
	float ym_per_pix = 3.6 / 422;
	float curve_radius;

	Mat coef_cr(3, 1, CV_32F);
	
	vector<Point2f> points_cvt;
	
	for(int i=0;i<points.size();i++) {
		points_cvt.push_back(Point2f(points[i].x * xm_per_pix, points[i].y * ym_per_pix));
	}

        coef_cr = polyfit(points_cvt);

	curve_radius = powf((1 + powf(2 * coef_cr.at<float>(2, 0) * 0 * xm_per_pix + coef_cr.at<float>(1, 0),2)), 1.5) / fabs(2 * coef_cr.at<float>(2, 0) + 0.000001);

	return curve_radius;
}

float calcCenterPos(Mat left_coef, Mat right_coef) {
	int car_position = 320;
	float ym_per_pix = 3.6 / 422;
	float xm_per_pix = 1.000 / 640;

	float left_y = left_coef.at<float>(0, 0);
	float right_y = right_coef.at<float>(0, 0);

	int lane_center_position = (left_y+ right_y) / 2;
	float center_position = (car_position - lane_center_position) * ym_per_pix;

	if(center_position < -0.1) cout<<"TURN LEFT"<<endl;
	else if(center_position > 0.1) cout<<"TURN RIGHT"<<endl;
	else cout<<"THE CAR IS GOING STRAIGHT."<<endl;
	
	return center_position;
}



int main(void) {
	BinaryParam bin_param;
	bin_param.l_thresh = 130;
	bin_param.max_val = 255;
	
	int last_leftx_base = 0;
	int last_rightx_base = 0;

	Mat left_coef(3, 1, CV_32F);
	Mat right_coef(3, 1, CV_32F);

	vector<Point2f> left_points;
	vector<Point2f> right_points;
	
	VideoCapture cap;
	cap.open("Demo1.mp4");
	
	for(;;) {
		Mat frame;
		cap>>frame;
	
		Mat exampleImg;
		resize(frame, exampleImg, Size(1280, 720));
	
		Point2f src[4];
		src[0] = Point(490, 630);
		src[1] = Point(810, 630);
		src[2] = Point(300, 720);
		src[3] = Point(1000, 720);
		
		Point2f dst[4]; 
		dst[0] = Point(180, 0);
		dst[1] = Point(exampleImg.cols - 180, 0);
		dst[2] = Point(250, exampleImg.rows);
		dst[3] = Point(exampleImg.cols - 250, exampleImg.rows);
	
		Mat exampleImg_unwarp = unwarp(exampleImg, src, dst);
		
		Mat exampleImg_pipe = pipeline(exampleImg_unwarp, bin_param);

		Mat l_pipe = Mat::zeros(Size(1280, 720), exampleImg_pipe.type());
		
		for(int j=0;j<720;j++) {
			for(int i=0;i<640;i++) {
				l_pipe.at<uchar>(j, i) = exampleImg_pipe.at<uchar>(j, i); 
			
			}
		}
	
		Mat r_pipe = Mat::zeros(Size(1280, 720), exampleImg_pipe.type());	
	
		for(int j=0;j<720;j++) {
			for(int i=640;i<1280;i++) {
				r_pipe.at<uchar>(j, i) = exampleImg_pipe.at<uchar>(j, i); 

			}
		}

		Mat left_hist(100, frame.cols, CV_8U);
		Mat right_hist(100, frame.cols, CV_8U);

		left_hist = getHistLane(l_pipe, last_leftx_base);
		right_hist = getHistLane(r_pipe, last_rightx_base);
	

/*
		if(last_leftx_base != 0) slidingWindow(exampleImg_pipe, l_pipe, last_leftx_base, left_points);
		if(last_rightx_base != 0) slidingWindow(exampleImg_pipe, r_pipe, last_rightx_base, right_points);
*/
		slidingWindow(exampleImg_pipe, l_pipe, last_leftx_base, left_points);
		slidingWindow(exampleImg_pipe, r_pipe, last_rightx_base, right_points);

		left_coef = polyfit(left_points);
		right_coef = polyfit(right_points);

		float left_cr = calcCurveRadius(left_points);
		float right_cr = calcCurveRadius(right_points);

		//cout << "left_cr: " << left_cr << endl;
		//cout << "right_cr: " << right_cr << endl;
		cout << "right_base: " << last_rightx_base << endl;

		//imshow("BEV", exampleImg_unwarp);
		imshow("PIPE WINDOW", exampleImg_pipe);
		imshow("left hist", left_hist);
		imshow("right hist", right_hist);
		imshow("FRAME", frame);
		waitKey(1);
	}
	return 0;
}



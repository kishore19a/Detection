#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>

#define PI 3.14159265

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
	float block_histogram_sum = 0.0;
	float block_histogram_modulus = 0.0;
	int arr_x[130][66];
	int arr_y[130][66];
	int arr_intensity[130][66];
	float arr_intensity_gradient[128][64];
	float arr_orientation[128][64];
	std::vector<float> histogram; histogram.resize(9,0.0);
	std::vector<vector<float> > cell;
	std::vector<float> block_histogram; block_histogram.resize(36,0.0);
	std::vector<vector<float> > block;

	if (argc != 2)
		{
			cout<<"Exactly 2 arguments should be there"<<endl;
			return -1;
		}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if(!image.data)
		{
			cout<<"Couldnot find or load the image"<<endl;
			return -1;
		}
    
    Size s = image.size();
    cout<<"Height = "<<s.height<<" , Width = "<<s.width<<endl;

    Mat gray_image;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    imwrite("Gray_Image_download.jpg", gray_image);

    Size sz(66,130);
    Mat resized_image;
    resize(gray_image,resized_image,sz);

    for (int x = 0; x < resized_image.rows; ++x)
	    {
	    	for (int y = 0; y < resized_image.cols; ++y)
		    	{
		    		arr_x[x][y] = 0;
		    		arr_y[x][y] = 0;
		    		Scalar pixel_intensity = resized_image.at<uchar>(x,y);
		    		arr_intensity[x][y] = pixel_intensity.val[0];
		    	}
	    }
     cout<<"resized_image.rows = "<<resized_image.rows<<" resized_image.cols = "<<resized_image.cols<<endl;

     for (int x = 1; x < resized_image.rows-1; ++x)
	    {
	    	for (int y = 1; y < resized_image.cols-1; ++y)
		    	{
		    		arr_x[x][y] =(-1*arr_intensity[x-1][y-1] + 0*arr_intensity[x-1][y] + 1*arr_intensity[x-1][y+1] +
		    		               -1*arr_intensity[x][y-1] + 0*arr_intensity[x][y] + 1*arr_intensity[x][y+1] +
		    		               -1*arr_intensity[x+1][y-1] + 0*arr_intensity[x+1][y] + 1*arr_intensity[x+1][y+1])/3;

		    		arr_y[x][y] = (1*arr_intensity[x-1][y-1] + 1*arr_intensity[x-1][y] + 1*arr_intensity[x-1][y+1] +
		    		               0*arr_intensity[x][y-1] + 0*arr_intensity[x][y] + 0*arr_intensity[x][y+1] +
		    		               -1*arr_intensity[x+1][y-1] + -1*arr_intensity[x+1][y] + -1*arr_intensity[x+1][y+1])/3;

		    		arr_intensity_gradient[x-1][y-1] = sqrt(arr_x[x][y]*arr_x[x][y] + arr_y[x][y]*arr_y[x][y]);
		    	    arr_orientation[x-1][y-1] = atan2(arr_y[x][y],arr_x[x][y]) * 180 / PI;
		    	    if(arr_orientation[x-1][y-1]<0) arr_orientation[x-1][y-1] += 180;
		    	}
	    }

	 Mat x_gradient = Mat::zeros(128, 64, CV_8UC1);
	 Mat y_gradient = Mat::zeros(128, 64, CV_8UC1);
	 Mat gradient_magnitude = Mat::zeros(128, 64, CV_8UC1);
	 Mat orientation_vector = Mat::zeros(128, 64, CV_8UC1);

     for (int x = 1; x < resized_image.rows-1; ++x)
	    {
	    	for (int y = 1; y < resized_image.cols-1; ++y)
		    	{
                    x_gradient.at<uchar>(x-1,y-1) = arr_x[x][y];
                    y_gradient.at<uchar>(x-1,y-1) = arr_y[x][y];
                    gradient_magnitude.at<uchar>(x-1,y-1) = arr_intensity_gradient[x][y];
                    orientation_vector.at<uchar>(x-1,y-1) = arr_orientation[x][y];
		    	}
	    }

	 for (int x = 0; x < resized_image.rows-2; x=x+8)
	 {
	 	for (int y = 0; y < resized_image.cols-2; y=y+8)
	 	{
	 		for (int i = x; i < x+8; ++i)
	 		{
	 			for (int j = y; j < y+8; ++j)
	 			{
	 				if (arr_orientation[i][j] >= 0 && arr_orientation[i][j] < 10)
	 				{
	 					histogram[0] = histogram[0] + arr_intensity_gradient[i][j];
	 				}
	 				if(arr_orientation[i][j] >= 10 && arr_orientation[i][j] < 30)
	 				{
                        histogram[0] = histogram[0] + (abs(arr_orientation[i][j]-30)/20)*(arr_intensity_gradient[i][j]);
                        histogram[1] = histogram[1] + (abs(arr_orientation[i][j]-10)/20)*(arr_intensity_gradient[i][j]);
	 				}
	 				if(arr_orientation[i][j] >= 30 && arr_orientation[i][j] < 50)
	 				{
                        histogram[1] = histogram[1] + (abs(arr_orientation[i][j]-50)/20)*(arr_intensity_gradient[i][j]);
                        histogram[2] = histogram[2] + (abs(arr_orientation[i][j]-30)/20)*(arr_intensity_gradient[i][j]);
	 				}
	 				if(arr_orientation[i][j] >= 50 && arr_orientation[i][j] < 70)
	 				{
                        histogram[2] = histogram[2] + (abs(arr_orientation[i][j]-70)/20)*(arr_intensity_gradient[i][j]);
                        histogram[3] = histogram[3] + (abs(arr_orientation[i][j]-50)/20)*(arr_intensity_gradient[i][j]);
	 				}
	 				if(arr_orientation[i][j] >= 70 && arr_orientation[i][j] < 90)
	 				{
                        histogram[3] = histogram[3] + (abs(arr_orientation[i][j]-90)/20)*(arr_intensity_gradient[i][j]);
                        histogram[4] = histogram[4] + (abs(arr_orientation[i][j]-70)/20)*(arr_intensity_gradient[i][j]);
	 				}
	 				if(arr_orientation[i][j] >= 90 && arr_orientation[i][j] < 110)
	 				{
                        histogram[4] = histogram[4] + (abs(arr_orientation[i][j]-110)/20)*(arr_intensity_gradient[i][j]);
                        histogram[5] = histogram[5] + (abs(arr_orientation[i][j]-90)/20)*(arr_intensity_gradient[i][j]);
	 				}
	 				if(arr_orientation[i][j] >= 110 && arr_orientation[i][j] < 130)
	 				{
                        histogram[5] = histogram[5] + (abs(arr_orientation[i][j]-130)/20)*(arr_intensity_gradient[i][j]);
                        histogram[6] = histogram[6] + (abs(arr_orientation[i][j]-110)/20)*(arr_intensity_gradient[i][j]);
	 				}
	 				if(arr_orientation[i][j] >= 130 && arr_orientation[i][j] < 150)
	 				{
                        histogram[6] = histogram[6] + (abs(arr_orientation[i][j]-150)/20)*(arr_intensity_gradient[i][j]);
                        histogram[7] = histogram[7] + (abs(arr_orientation[i][j]-130)/20)*(arr_intensity_gradient[i][j]);
	 				}
	 				if(arr_orientation[i][j] >= 150 && arr_orientation[i][j] < 170)
	 				{
                        histogram[7] = histogram[7] + (abs(arr_orientation[i][j]-170)/20)*(arr_intensity_gradient[i][j]);
                        histogram[8] = histogram[8] + (abs(arr_orientation[i][j]-150)/20)*(arr_intensity_gradient[i][j]);
	 				}
	 				if(arr_orientation[i][j] >= 170 && arr_orientation[i][j] < 180)
	 				{
                        histogram[8] = histogram[8] + (arr_intensity_gradient[i][j]);
	 				}
	 			}
	 		}
	 		cell.push_back(histogram);
	 	}
	 }

	 // for (int x = 0; x < 15; ++x)
	 // {
	 // 	for (int y = 0; y < 7; ++y)
	 // 	{
	 // 	    for (int i = 0; i < 36; ++i)
	 // 	    {
	 // 	    	if(i>=0 && i<9) block_histogram[i] = cell[x*16+y].histogram[i];
  //               if(i>=9 && i<18) block_histogram[i] = cell[x*16+y+1].histogram[i-9];
  //               if(i>=18 && i<27) block_histogram[i] = cell[(x+1)*16+y].histogram[i-18];
  //               if(i>=18 && i<36) block_histogram[i] = cell[(x+1)*16+y+1].histogram[i-27];
  //               block_histogram_sum = block_histogram_sum + block_histogram[i]*block_histogram[i];
	 // 	    }
	 // 	    block_histogram_modulus = sqrt(block_histogram_sum);
	 // 	    for (int i = 0; i < 36; ++i)
	 // 	    {
	 // 	    	block_histogram[i] = block_histogram[i]/block_histogram_modulus;
	 // 	    }
	 // 	    block.push_back(block_histogram);
	 // 	}
	 // }

	namedWindow("Original Image", WINDOW_AUTOSIZE);
	namedWindow("GrayScale Image", WINDOW_AUTOSIZE);
	namedWindow("Resized Image", WINDOW_AUTOSIZE);
	namedWindow("X Gradient", WINDOW_AUTOSIZE);
	namedWindow("Y Gradient", WINDOW_AUTOSIZE);
	namedWindow("Gradient Magnitude", WINDOW_NORMAL);
	namedWindow("Orientation", WINDOW_NORMAL);

	imshow("Original Image", image);
	imshow("GrayScale Image", gray_image);
	imshow("Resized Image", resized_image);
	imshow("X Gradient", x_gradient);
	imshow("Y Gradient", y_gradient);
	imshow("Gradient Magnitude", gradient_magnitude);
	imshow("Orientation", orientation_vector);

	waitKey(0);
	return 0;
}


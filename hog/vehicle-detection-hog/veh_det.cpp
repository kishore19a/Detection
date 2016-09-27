#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>
#include <tr1/memory>

// includes for file_exists and files_in_directory functions
#ifndef __linux
#include <io.h> 
#define access _access_s
#else
#include <unistd.h>
#include <memory>
#endif

#define POSITIVE_TRAINING_SET_PATH "DATASET/POSITIVE"
#define NEGATIVE_TRAINING_SET_PATH "DATASET/NEGATIVE"
#define WINDOW_NAME "WINDOW"
#define TRAFFIC_VIDEO_FILE "video.mp4"
#define TRAINED_SVM "vehicle_detector.yml"
#define	IMAGE_SIZE Size(40, 40) 

using namespace cv;
using namespace cv::ml;
using namespace std;

bool file_exists(const string &file);
void load_images(string directory, vector<Mat>& image_list);
vector<string> files_in_directory(string directory);

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);
void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size);
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size);
void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels);
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);
void test_it(const Size & size);


int main(int argc, char** argv)
{
	if (!file_exists(TRAINED_SVM)) {
        

		vector< Mat > pos_lst;
		vector< Mat > full_neg_lst;
		vector< Mat > neg_lst;
		vector< Mat > gradient_lst;
		vector< int > labels;

		load_images(POSITIVE_TRAINING_SET_PATH, pos_lst);
		labels.assign(pos_lst.size(), +1);
		
		load_images(NEGATIVE_TRAINING_SET_PATH, full_neg_lst);
		labels.insert(labels.end(), full_neg_lst.size(), -1);

		compute_hog(pos_lst, gradient_lst, IMAGE_SIZE);
		compute_hog(full_neg_lst, gradient_lst, IMAGE_SIZE);

		// train_svm(gradient_lst, labels);

	}

	// test_it(IMAGE_SIZE);
	return 0;
}

bool file_exists(const string &file)
{
	return access(file.c_str(), 0) == 0;
}

vector<string> files_in_directory(string directory)
{
	vector<string> files;
	char buf[256];
	string command;

#ifdef __linux__ 
	command = "ls " + directory;
    std::tr1::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);

	char cwd[256];
	getcwd(cwd, sizeof(cwd));

	while (!feof(pipe.get()))
		if (fgets(buf, 256, pipe.get()) != NULL) {
			string file(cwd);
			file.append("/");
			file.append(directory);
			file.append("/");
			file.append(buf);
			file.pop_back();
			files.push_back(file);
		}
#else
	command = "dir /b /s " + directory;
	FILE* pipe = NULL;

	if (pipe = _popen(command.c_str(), "rt"))
		while (!feof(pipe))
			if (fgets(buf, 256, pipe) != NULL) {
				string file(buf);
				file.pop_back();
				files.push_back(file);
			}
	_pclose(pipe);
#endif
	return files;
}

void load_images(string directory, vector<Mat>& image_list) {

	Mat img;
	vector<string> files;
	files = files_in_directory(directory);

	for (int i = 0; i < files.size(); ++i) {

		img = imread(files.at(i));
		if (img.empty())
			continue;
#ifdef _DEBUG
		imshow("image", img);
		waitKey(10);
#endif
		resize(img, img, IMAGE_SIZE);
		image_list.push_back(img.clone());
	}
	cout<<image_list.size()<<endl;
}

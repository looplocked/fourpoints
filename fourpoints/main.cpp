#include<opencv2\highgui.hpp>
#include<iostream>
#include "LineSegmentDetector.h"
using namespace std;
using namespace cv;
typedef std::vector<std::pair<double, int> > sortType;


//程序入口函数
//输入 视频路径；
//输出 平面角点
bool findPrimaryAngle(std::vector<cv::Vec4f>& lines);
bool selectPoints(std::vector<cv::Point>& Points);

int main(int argc, char** argv)
{
	//设置路径
	std::string path = ".\\项目要求\\cube1.mp4";
	if (argc >= 2)
	{
		//可以cmd 输入路径
		path = argv[1];
	}

	//打开视频
	VideoCapture m_capture;
	m_capture.open("cube1.mp4");

	if (!m_capture.isOpened())
	{
		cout << "No camera or video input!\n" << endl;
		return -1;
	}
	//******************

	//LSR直线检测算法初始化
	cv::Ptr<cv_::LineSegmentDetector> ls;
#if 0
	ls = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);//或者两种LSD算法，这边用的是standard的
#else
	ls = cv_::createLineSegmentDetector(cv_::LSD_REFINE_NONE);
#endif

	//
	int count = 0;
	cv::Mat frame, mask, gray;
	cv::Mat backframe;

	while (m_capture.grab())
	{
		count++;
		m_capture >> frame;

		if (frame.cols == 0 || frame.rows == 0)
		{
			break;
		}

		resize(frame, frame, Size(800, 600));//输入视频调整为800*800大小
		cvtColor(frame, gray, CV_RGB2GRAY);//图像灰度化

										   //二值化
		cv::Mat img_binary = gray.clone();

		//图像处理形态学计算
		threshold(img_binary, img_binary, 240, 255, CV_THRESH_BINARY);
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(16, 16));
		morphologyEx(img_binary, img_binary, cv::MORPH_OPEN, element);


		cv::Mat showbinary = img_binary.clone();
		cv::resize(showbinary, showbinary, cv::Size(800, 800));

		//寻找合适的联通区域
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(img_binary, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		int maxArea = 0;
		cv::Rect r1;
		CvBox2D r2;

		std::vector<cv::Rect> rects;
		std::vector<cv::Rect> rects_all;
		for (int i = 0; i < contours.size(); i++)
		{
			maxArea = contours[i].size();
			r2 = cv::minAreaRect(contours[i]);
			r1 = cv::boundingRect(contours[i]);
			rects_all.push_back(r1);
			if (r1.area()<10000 || r1.area()>50000)
			{
				continue;
			}
			std::cout << r1.area() << " ";

			rects.push_back(r1);
		}
		std::cout << std::endl;
		//***************************


		//
		int max_dis = 999;
		cv::Rect r(0, 0, 0, 0);
		cv::Point P1, P2;
		P1.x = frame.cols*0.5;
		P1.y = frame.rows*0.5;

		for (int i = 0; i < rects.size(); i++)
		{
			P2.x = rects[i].x + 0.5*rects[i].width;
			P2.y = rects[i].y + 0.5*rects[i].height;

			double distance;
			distance = powf((P1.x - P2.x), 2) + powf((P1.y - P2.y), 2);
			distance = sqrtf(distance);

			if (distance < max_dis)
			{
				max_dis = distance;
				r = rects[i];
			}
		}

		int max_all = 0;
		if (r.area() == 0)
		{
			for (int i = 0; i < rects_all.size(); i++)
			{
				if (rects_all[i].area()>max_all)
				{
					max_all = rects_all[i].area();
					r = rects_all[i];
				}
			}
		}


		//感兴趣区域扩展
		if (r.area() == 0)
		{
			r = cv::Rect(0, 0, 799, 799);
		}
		else
		{
			r.x -= 10;
			r.y -= 10;
			r.width += 20;
			r.height += 20;
		}

		std::vector<cv::Vec4f> lines_std;
		lines_std.reserve(1000);
		ls->detect(gray(r), lines_std);//这里把检测到的直线线段都存入了lines_std中，4个float的值，分别为起止点的坐标

									   //去除干扰线段
		findPrimaryAngle(lines_std);
		//*****************************

		cv::Mat drawImg = frame.clone();
		cv::Mat drawImg2 = frame.clone();
		//ls->drawSegments(drawImg, lines_std);
		//imshow("2", drawImg);

		std::vector<cv::Point> Points;
		for (int i = 0; i < lines_std.size(); i++)
		{
			cv::Point p1, p2;
			p1 = cv::Point(lines_std[i][0] + r.x, lines_std[i][1] + r.y);
			p2 = cv::Point(lines_std[i][2] + r.x, lines_std[i][3] + r.y);

			Points.push_back(p1);
			Points.push_back(p2);
		}

		//条件滤波 暂时关闭
		//selectPoints(Points);
		//******************************

		//画图
		for (int i = 0; i < Points.size(); i++)
		{
			circle(drawImg2, Points[i], 5, Scalar(0, 0, 255), 2, 18);
		}
		//**********************

		imshow("结果", drawImg2);
		//可以控制视频的播放速度
		cv::waitKey(10);

		frame.release();
	}
}

bool SortFaster(sortType& dists, int Top_num)
{
	if (dists.size() < Top_num || Top_num == 0)
	{
		return false;
	}

	int length_dists = dists.size();

	int num = 0;

	while (num<Top_num)
	{
		for (int i = length_dists - 1; i > num; i--)
		{
			if (dists[i].first > dists[i - 1].first)
			{
				std::pair<double, int> temp;
				temp = dists[i];
				dists[i] = dists[i - 1];
				dists[i - 1] = temp;
			}
		}
		num++;
	}

	return true;
}

bool findPrimaryAngle(std::vector<cv::Vec4f>& lines)
{
	if (lines.empty())
	{
		return false;
	}

	std::vector<cv::Vec4f> linsNew;

	int count = 0;
	sortType distlist;
	for (auto i : lines)
	{
		double dist = norm(cv::Point(i[0], i[1]) - cv::Point(i[2], i[3]));
		distlist.push_back(std::pair<double, int>(dist, count++));
	}

	//-----------------------------------------------
	int top_nun = 2;//获取最长直线的个数
	SortFaster(distlist, top_nun);

	int length = MIN(distlist.size(), top_nun);

	int mean_lenth = 0;
	for (int i = 0; i < length; i++)
	{
		//linsNew.push_back(lines[distlist[i].second]);
		mean_lenth += distlist[i].first;
	}

	mean_lenth /= length;

	std::cout << "m:  " << mean_lenth << std::endl;

	for (int i = 0; i < length; i++)
	{
		if (distlist[i].first>0.8*mean_lenth)
		{
			linsNew.push_back(lines[distlist[i].second]);
		}
	}

	if (linsNew.size() <= 4)
	{
		lines.clear();
		lines = linsNew;
		return true;
	}

	double min_angle = 1000000;

	int pos1, pos2;

	for (int i = 0; i < linsNew.size() - 1; i++)
	{
		double angle = double(linsNew[i][1] - linsNew[i][3]) / double(linsNew[i][0] - linsNew[i][2]);

		for (int j = i + 1; j < linsNew.size(); j++)
		{
			double angle2 = double(linsNew[j][1] - linsNew[j][3]) / double(linsNew[j][0] - linsNew[j][2]);

			double dist = abs(angle2 - angle);
			double lenth = norm(cv::Point(linsNew[i][2] - linsNew[i][3]) - cv::Point(linsNew[j][2] - linsNew[j][3]));

			//std::cout << lenth << "ddfa" << std::endl;
			if (dist < min_angle&&length>180)
			{
				min_angle = dist;
				pos1 = i;
				pos2 = j;
			}
		}
	}


	lines.clear();
	lines.push_back(linsNew[pos1]);
	lines.push_back(linsNew[pos2]);

	return true;
}

bool selectPoints(std::vector<cv::Point>& Points)
{
	std::vector<int> dels;
	std::vector<cv::Point> Points_out;
	for (int i = 0; i < Points.size() / 2 - 1; i++)
	{
		bool isF = true, isS = true;

		for (int k = 0; k < dels.size(); k++)
		{
			if (2 * i == dels[k])
			{
				isF = false;
			}

			if (2 * i + 1 == dels[k])
			{
				isS = false;
			}
		}

		for (int j = 2 * i + 2; j < Points.size(); j++)
		{
			if (isF)
			{
				double dist1 = norm(Points[i * 2] - Points[j]);

				if (dist1<50)
				{
					double l1 = norm(Points[i * 2] - Points[i * 2 + 1]);
					double l2;
					if (j % 2 == 0)
					{
						l2 = norm(Points[j] - Points[j + 1]);
					}
					else
					{
						l2 = norm(Points[j] - Points[j - 1]);
					}

					if (l1 >= l2)
					{
						Points_out.push_back(Points[i * 2]);
						dels.push_back(j);
					}
				}
				else
				{
					Points_out.push_back(Points[i * 2]);
				}

			}

			if (isS)
			{
				double dist2 = norm(Points[i * 2 + 1] - Points[j]);

				if (dist2 < 50)
				{
					double l1 = norm(Points[i * 2] - Points[i * 2 + 1]);
					double l2;
					if (j % 2 == 0)
					{
						l2 = norm(Points[j] - Points[j + 1]);
					}
					else
					{
						l2 = norm(Points[j] - Points[j - 1]);
					}

					if (l1 >= l2)
					{
						Points_out.push_back(Points[i * 2 + 1]);
						dels.push_back(j);
					}
				}
				else
				{
					Points_out.push_back(Points[i * 2 + 1]);
				}
			}
		}
	}
	Points = Points_out;

	return true;
}



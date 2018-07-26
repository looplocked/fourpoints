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
void enableClockWise(std::vector<cv::Point>& Points);
void transPoints(std::vector<cv::Point>& Points, int index);

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
	//m_capture.open("cube1.mp4");
	m_capture.open(0);

	if (!m_capture.isOpened())
	{
		cout << "No camera or video input!\n" << endl;
		return -1;
	}
	//******************

	//LSR直线检测算法初始化
	cv::Ptr<cv_::LineSegmentDetector> ls;
#if 0
	ls = cv_::createLineSegmentDetector(cv_::LSD_REFINE_STD);//或者两种LSD算法，这边用的是standard的
#else
	ls = cv_::createLineSegmentDetector(cv_::LSD_REFINE_NONE);
#endif

	//
	int count = 0;
	cv::Mat frame, mask, gray;
	cv::Mat backframe;

	std::vector<cv::Point> prePoints = { Point(220, 140), Point(420, 140), Point(220, 340), Point(420, 340) };

	while (m_capture.grab())
	{
		count++;
		m_capture >> frame;

		if (frame.cols == 0 || frame.rows == 0)
		{
			break;
		}

		resize(frame, frame, Size(640, 480));//输入视频调整为800*800大小
		cvtColor(frame, gray, CV_RGB2GRAY);//图像灰度化

										   //二值化
		cv::Mat img_binary = gray.clone();

		//图像处理形态学计算
		threshold(img_binary, img_binary, 235, 255, CV_THRESH_BINARY);
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(16, 16));
		morphologyEx(img_binary, img_binary, cv::MORPH_OPEN, element);


		cv::Mat showbinary = img_binary.clone();
		cv::resize(showbinary, showbinary, cv::Size(640, 480));

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
			r = cv::Rect(0, 0, 639, 479);
		}
		else
		{
			r.x = r.x >= 10 ? r.x - 10 : 0;
			r.y = r.y >= 10 ? r.y - 10 : 0;
			r.width = r.x + r.width <= 620 ? r.width + 20 : 640 - r.x;
			r.height = r.y + r.height <= 460 ? r.height + 20 : 460 - r.y;
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

		int index = 0;
		double mindist = 1000;
		for (int i = 0; i < 4; i++) {
			double d = norm(prePoints[0] - Points[i]);
			if (d < mindist) {
				index = i;
				mindist = d;
			}
		}

		transPoints(Points, index);

		std::vector<cv::Point> prePoints = Points;

		//条件滤波 暂时关闭
		//selectPoints(Points);
		//******************************

		//画图
		int font_face = cv::FONT_HERSHEY_COMPLEX;
		double font_scale = 0.5;
		int thickness = 0.2;
		for (int i = 0; i < Points.size(); i++)
		{
			circle(drawImg2, Points[i], 5, Scalar(0, 0, 255), 2, 18);
			string text = "point" + to_string(i);
			putText(drawImg2, text, Points[i], font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);
			int j = i == Points.size() - 1 ? 0 : i + 1;
			line(drawImg2, Points[i], Points[j], Scalar(255, 0, 0), 2, 8);
		}

		line(drawImg2, cv::Point(r.x, r.y), cv::Point(r.x + r.width, r.y), Scalar(0, 166, 0), 2, 8);
		line(drawImg2, cv::Point(r.x+r.width, r.y), cv::Point(r.x + r.width, r.y + r.height), Scalar(0, 166, 0), 2, 8);
		line(drawImg2, cv::Point(r.x+r.width, r.y+r.height), cv::Point(r.x, r.y + r.height), Scalar(0, 166, 0), 2, 8);
		line(drawImg2, cv::Point(r.x, r.y+r.height), cv::Point(r.x, r.y), Scalar(0, 166, 0), 2, 8);
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

bool  needReverse(cv::Vec4f line1, cv::Vec4f line2)
{
	Point2f ptr1(line1[2] - line1[0], line1[3] - line1[1]);
	Point2f ptr2(line2[2] - line2[0], line2[3] - line2[1]);
	double con = ptr1.x * ptr2.x + ptr1.y * ptr2.y;
	return con > 0;
}

void enableClockWise(std::vector<cv::Point>& Points)
{
	Point vec1(Points[1].x - Points[0].x, Points[1].y - Points[0].y);
	Point vec2(Points[2].x - Points[1].x, Points[2].y - Points[1].y);
	if (vec1.x * vec2.y - vec1.y * vec2.x < 0) {
		Point temp(Points[1].x, Points[1].y);
		Points[1] = Points[3];
		Points[3] = Points[1];
	}
}

void transPoints(std::vector<cv::Point>& Points, int k)
{
	k = k % Points.size();
	reverse(begin(Points), begin(Points) + Points.size() - k);
	reverse(begin(Points) + Points.size() - k, Points.end());
	reverse(begin(Points), end(Points));
}


bool findPrimaryAngle(std::vector<cv::Vec4f>& lines)
{
	if (lines.empty())
	{
		return false;
	}
	int linenum = lines.size();

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

	for (int i = 0; i < linenum; i++)
	{
		if (distlist[i].first>0.8*mean_lenth)
		{
			linsNew.push_back(lines[distlist[i].second]);
		}
	}

	if (linsNew.size() <= 1)
	{
		lines.clear();
		lines = linsNew;
		return true;
	}
	else if (linsNew.size() == 2) {
		if (needReverse(linsNew[0], linsNew[1])) {
			cv::Vec4f temp(linsNew[1][2], linsNew[1][3], linsNew[1][0], linsNew[1][1]);
			linsNew[1] = temp;
		}
		lines.clear();
		lines = linsNew;
		return true;
	}

	double min_angle = 1000000;

	int pos1 = 0, pos2 = 2;

	for (int i = 0; i < linsNew.size() - 1; i++)
	{
		double angle = double(linsNew[i][1] - linsNew[i][3]) / double(linsNew[i][0] - linsNew[i][2]);
		angle = abs(angle);

		for (int j = i + 1; j < linsNew.size(); j++)
		{
			double angle2 = double(linsNew[j][1] - linsNew[j][3]) / double(linsNew[j][0] - linsNew[j][2]);
			angle2 = abs(angle2);

			double dist = abs(angle2 - angle);
			double x1 = (linsNew[i][0] + linsNew[i][2]) / 2;
			double y1 = (linsNew[i][1] + linsNew[i][3]) / 2;
			double x2 = (linsNew[j][0] + linsNew[j][2]) / 2;
			double y2 = (linsNew[j][1] + linsNew[j][3]) / 2;
			double linesDis = norm(cv::Point(x1, y1) - cv::Point(x2, y2));

			if (dist < min_angle && linesDis > 0.1*mean_lenth) //平行且不重合的两条线
			{
				min_angle = dist;
				pos1 = i;
				pos2 = j;
			}
		}
	}

	if (needReverse(linsNew[pos1], linsNew[pos2])) {
		cv::Vec4f temp(linsNew[pos2][2], linsNew[pos2][3], linsNew[pos2][0], linsNew[pos2][1]);
		linsNew[pos2] = temp;
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



#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>


namespace cv_
{

	class LineSegmentDetector : public cv::Algorithm
	{
	public:

		virtual void detect(cv::InputArray _image, cv::OutputArray _lines,
			cv::OutputArray width = cv::noArray(), cv::OutputArray prec = cv::noArray(),
			cv::OutputArray nfa = cv::noArray()) = 0;

		virtual void drawSegments(cv::InputOutputArray _image, cv::InputArray lines) = 0;
		virtual int compareSegments(const cv::Size& size, cv::InputArray lines1, cv::InputArray lines2, cv::InputOutputArray _image = cv::noArray()) = 0;
		virtual ~LineSegmentDetector() {}
	};

	enum LineSegmentDetectorModes {
		LSD_REFINE_NONE = 0, //!< No refinement applied
		LSD_REFINE_STD = 1, //!< Standard refinement is applied. E.g. breaking arches into smaller straighter line approximations.
		LSD_REFINE_ADV = 2  //!< Advanced refinement. Number of false alarms is calculated, lines are
							//!< refined through increase of precision, decrement in size, etc.
	};

	cv::Ptr<cv_::LineSegmentDetector> createLineSegmentDetector(
		int _refine = 1, double _scale = 0.8,
		double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5,
		double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024);

}
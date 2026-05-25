#ifndef UAS_PCD_25_H
#define UAS_PCD_25_H

#include <opencv2/opencv.hpp>
#include <string>

// Helper: tampilkan Before & After berdampingan
void showBeforeAfter(const cv::Mat& before, const cv::Mat& after, const std::string& title);

void tampilkanCitra(const cv::Mat& img);
void aksesDataCitra(const cv::Mat& img);
cv::Mat ubahKeGrayscale(const cv::Mat& img);
cv::Mat ubahKeNegative(const cv::Mat& img);
cv::Mat tambahBrightness(const cv::Mat& img, int value);
void tampilkanHistogram(const cv::Mat& img);
cv::Mat equalizeHistGray(const cv::Mat& img);
cv::Mat equalizeHistColor(const cv::Mat& img);
void histogramMatching(std::string inputPath, std::string referencePath);
cv::Mat applyMedianBlur(const cv::Mat& img);
cv::Mat applyBilateralBlur(const cv::Mat& img);
cv::Mat applySharpening(const cv::Mat& img);
cv::Mat applyLaplace(const cv::Mat& img);
cv::Mat applySobelCanny(const cv::Mat& img);
cv::Mat applyPrewitt(const cv::Mat& img);
cv::Mat applyThreshold(const cv::Mat& img);
cv::Mat applyAdaptiveThreshold(const cv::Mat& img);
cv::Mat regionGrowing(const cv::Mat& img, cv::Point seed);
cv::Mat splitAndMerge(const cv::Mat& img);
cv::Mat clusteringSegmentation(const cv::Mat& img);
cv::Mat applyDilateErode(const cv::Mat& img);
cv::Mat applyOpenClose(const cv::Mat& img);
cv::Mat applyHitOrMiss(const cv::Mat& img);
cv::Mat compressImage(const cv::Mat& img);

// Menu 25-28
void fidelityKriteria(const cv::Mat& original, const cv::Mat& compressed);
cv::Mat pengenalanFeature(const cv::Mat& img);
cv::Mat patternRecognition(const cv::Mat& img);
cv::Mat rekonstruksiCitra(const cv::Mat& img);

#endif
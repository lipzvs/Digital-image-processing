#include "UAS_PCD_25.h"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace cv;
using namespace std;

// =============================================
// Helper: tampilkan Before & After berdampingan
// =============================================
void showBeforeAfter(const Mat& before, const Mat& after, const string& title) {
    // Samakan tinggi kedua gambar
    Mat leftImg, rightImg;

    // Kalau after grayscale, convert ke BGR supaya bisa digabung
    if (before.channels() == 3 && after.channels() == 1) {
        cvtColor(after, rightImg, COLOR_GRAY2BGR);
    }
    else if (before.channels() == 1 && after.channels() == 3) {
        cvtColor(before, leftImg, COLOR_GRAY2BGR);
        rightImg = after.clone();
    }
    else {
        rightImg = after.clone();
    }

    if (leftImg.empty()) leftImg = before.clone();

    // Resize ke tinggi yang sama
    int targetH = max(leftImg.rows, rightImg.rows);
    if (leftImg.rows != targetH) resize(leftImg, leftImg, Size(leftImg.cols * targetH / leftImg.rows, targetH));
    if (rightImg.rows != targetH) resize(rightImg, rightImg, Size(rightImg.cols * targetH / rightImg.rows, targetH));

    // Tambahkan label "Before" dan "After"
    Mat labelBefore = Mat::zeros(30, leftImg.cols, CV_8UC3);
    Mat labelAfter = Mat::zeros(30, rightImg.cols, CV_8UC3);
    putText(labelBefore, "Before", Point(10, 22), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    putText(labelAfter, "After", Point(10, 22), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 100), 2);

    Mat leftFull, rightFull;
    vconcat(labelBefore, leftImg, leftFull);
    vconcat(labelAfter, rightImg, rightFull);

    // Garis pemisah vertikal
    Mat divider = Mat::zeros(leftFull.rows, 3, CV_8UC3);
    divider.setTo(Scalar(200, 200, 200));

    // Gabung horizontal
    Mat combined;
    hconcat(vector<Mat>{leftFull, divider, rightFull}, combined);

    imshow(title + " | Before vs After", combined);
}

void tampilkanCitra(const Mat& img) {
    imshow("Citra Digital", img);
    waitKey(0);
    destroyAllWindows();
}

void aksesDataCitra(const Mat& img) {
    cout << "Ukuran: " << img.cols << " x " << img.rows << endl;
    cout << "Channels: " << img.channels() << endl;
    cout << "Tipe: " << img.type() << endl;
}

Mat ubahKeGrayscale(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    return gray;
}

Mat ubahKeNegative(const Mat& img) {
    Mat negatif;
    bitwise_not(img, negatif);
    return negatif;
}

Mat tambahBrightness(const Mat& img, int value) {
    Mat terang;
    img.convertTo(terang, -1, 1, value);
    return terang;
}

void tampilkanHistogram(const Mat& img) {
    vector<Mat> bgr_planes;
    split(img, bgr_planes);
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2);
    }
    imshow("Histogram", histImage);
    waitKey(0);
    destroyAllWindows();
}

Mat equalizeHistGray(const Mat& img) {
    Mat gray = ubahKeGrayscale(img), result;
    equalizeHist(gray, result);
    return result;
}

Mat equalizeHistColor(const Mat& img) {
    Mat ycrcb;
    cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb, channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, ycrcb);
    Mat result;
    cvtColor(ycrcb, result, COLOR_YCrCb2BGR);
    return result;
}

// =============================================
// Menu 9: Histogram Matching (FIXED)
// Memetakan distribusi warna gambar input ke gambar referensi
// Hasilnya: harimau berwarna sesuai palet warna referensi (hijau)
// =============================================
void histogramMatching(string inputPath, string referencePath) {
    Mat input = imread(inputPath);
    Mat reference = imread(referencePath);
    if (input.empty() || reference.empty()) {
        cout << "Gagal membuka citra input atau referensi." << endl;
        return;
    }

    vector<Mat> inputChannels, refChannels, resultChannels;
    split(input, inputChannels);
    split(reference, refChannels);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    int ch = 0;

    for (int c = 0; c < 3; c++) {
        Mat inputHist, refHist;
        calcHist(&inputChannels[c], 1, &ch, Mat(), inputHist, 1, &histSize, histRange);
        calcHist(&refChannels[c], 1, &ch, Mat(), refHist, 1, &histSize, histRange);

        // Hitung CDF input
        Mat cdfInput(256, 1, CV_64F);
        cdfInput.at<double>(0) = inputHist.at<float>(0);
        for (int i = 1; i < 256; i++)
            cdfInput.at<double>(i) = cdfInput.at<double>(i - 1) + inputHist.at<float>(i);

        // Hitung CDF referensi
        Mat cdfRef(256, 1, CV_64F);
        cdfRef.at<double>(0) = refHist.at<float>(0);
        for (int i = 1; i < 256; i++)
            cdfRef.at<double>(i) = cdfRef.at<double>(i - 1) + refHist.at<float>(i);

        // Normalisasi CDF ke [0,1]
        double totalInput = cdfInput.at<double>(255);
        double totalRef = cdfRef.at<double>(255);
        for (int i = 0; i < 256; i++) {
            cdfInput.at<double>(i) /= totalInput;
            cdfRef.at<double>(i) /= totalRef;
        }

        // Buat LUT: untuk setiap nilai pixel input, cari nilai di referensi yang CDF-nya paling dekat
        uchar LUT[256];
        for (int i = 0; i < 256; i++) {
            double val = cdfInput.at<double>(i);
            int closest = 0;
            double minDiff = 1.0;
            for (int j = 0; j < 256; j++) {
                double diff = abs(val - cdfRef.at<double>(j));
                if (diff < minDiff) {
                    minDiff = diff;
                    closest = j;
                }
            }
            LUT[i] = (uchar)closest;
        }

        // Terapkan LUT
        Mat resultChannel(inputChannels[c].size(), inputChannels[c].type());
        for (int i = 0; i < inputChannels[c].rows; i++)
            for (int j = 0; j < inputChannels[c].cols; j++)
                resultChannel.at<uchar>(i, j) = LUT[inputChannels[c].at<uchar>(i, j)];

        resultChannels.push_back(resultChannel);
    }

    Mat result;
    merge(resultChannels, result);

    // Before & After
    imshow("Input (Gambar Asli)", input);
    imshow("Referensi (Gambar Warna Target)", reference);
    showBeforeAfter(input, result, "Histogram Matching");

    waitKey(0);
    destroyAllWindows();
}

Mat applyMedianBlur(const Mat& img) {
    Mat result;
    medianBlur(img, result, 5);
    return result;
}

Mat applyBilateralBlur(const Mat& img) {
    Mat result;
    bilateralFilter(img, result, 9, 75, 75);
    return result;
}

Mat applySharpening(const Mat& img) {
    Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    Mat result;
    filter2D(img, result, img.depth(), kernel);
    return result;
}

Mat applyLaplace(const Mat& img) {
    Mat gray, laplace;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Laplacian(gray, laplace, CV_16S);
    convertScaleAbs(laplace, laplace);
    return laplace;
}

Mat applySobelCanny(const Mat& img) {
    Mat gray, edges;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Canny(gray, edges, 100, 200);
    return edges;
}

Mat applyPrewitt(const Mat& img) {
    Mat gray, gx, gy, prewitt;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat kernelx = (Mat_<char>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    Mat kernely = (Mat_<char>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    filter2D(gray, gx, gray.depth(), kernelx);
    filter2D(gray, gy, gray.depth(), kernely);
    addWeighted(gx, 0.5, gy, 0.5, 0, prewitt);
    return prewitt;
}

Mat applyThreshold(const Mat& img) {
    Mat gray, result;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, result, 128, 255, THRESH_BINARY);
    return result;
}

Mat applyAdaptiveThreshold(const Mat& img) {
    Mat gray, result;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    adaptiveThreshold(gray, result, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    return result;
}

Mat regionGrowing(const Mat& img, Point seed) {
    Mat gray = ubahKeGrayscale(img);
    Mat mask = Mat::zeros(gray.rows + 2, gray.cols + 2, CV_8UC1);
    floodFill(gray, mask, seed, Scalar(255), 0, Scalar(10), Scalar(10), 4);
    return gray;
}

Mat splitAndMerge(const Mat& img) {
    return applyThreshold(img);
}

Mat clusteringSegmentation(const Mat& img) {
    Mat samples(img.rows * img.cols, 3, CV_32F);
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            for (int z = 0; z < 3; z++)
                samples.at<float>(y + x * img.rows, z) = img.at<Vec3b>(y, x)[z];
    Mat labels, centers;
    kmeans(samples, 3, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
    Mat new_image(img.size(), img.type());
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * img.rows, 0);
            for (int z = 0; z < 3; z++)
                new_image.at<Vec3b>(y, x)[z] = centers.at<float>(cluster_idx, z);
        }
    return new_image;
}

Mat applyDilateErode(const Mat& img) {
    Mat result;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(img, result, kernel);
    erode(result, result, kernel);
    return result;
}

Mat applyOpenClose(const Mat& img) {
    Mat result;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(img, result, MORPH_OPEN, kernel);
    morphologyEx(result, result, MORPH_CLOSE, kernel);
    return result;
}

Mat applyHitOrMiss(const Mat& img) {
    Mat gray = ubahKeGrayscale(img), bin;
    threshold(gray, bin, 128, 255, THRESH_BINARY);
    Mat kernel = (Mat_<char>(3, 3) << 0, 1, 0, 1, -1, 1, 0, 1, 0);
    Mat result;
    morphologyEx(bin, result, MORPH_HITMISS, kernel);
    return result;
}

Mat compressImage(const Mat& img) {
    vector<uchar> buf;
    vector<int> params = { IMWRITE_JPEG_QUALITY, 20 };
    imencode(".jpg", img, buf, params);
    Mat result = imdecode(buf, IMREAD_COLOR);
    return result;
}

// =============================================
// Menu 25: Fidelity Kriteria
// =============================================
void fidelityKriteria(const Mat& original, const Mat& compressed) {
    Mat orig_gray, comp_gray;
    cvtColor(original, orig_gray, COLOR_BGR2GRAY);
    cvtColor(compressed, comp_gray, COLOR_BGR2GRAY);

    orig_gray.convertTo(orig_gray, CV_64F);
    comp_gray.convertTo(comp_gray, CV_64F);

    Mat diff;
    absdiff(orig_gray, comp_gray, diff);
    diff = diff.mul(diff);
    double mse = sum(diff)[0] / (double)(orig_gray.rows * orig_gray.cols);

    double psnr = (mse > 0) ? 10.0 * log10((255.0 * 255.0) / mse) : 100.0;

    double C1 = 6.5025, C2 = 58.5225;
    double mu1 = mean(orig_gray)[0];
    double mu2 = mean(comp_gray)[0];
    Mat orig_sq = orig_gray.mul(orig_gray);
    Mat comp_sq = comp_gray.mul(comp_gray);
    Mat orig_comp = orig_gray.mul(comp_gray);
    double sigma1_sq = mean(orig_sq)[0] - mu1 * mu1;
    double sigma2_sq = mean(comp_sq)[0] - mu2 * mu2;
    double sigma12 = mean(orig_comp)[0] - mu1 * mu2;
    double ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2))
        / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2));

    cout << "\n=== Fidelity Kriteria ===" << endl;
    cout << fixed << setprecision(4);
    cout << "MSE  : " << mse << endl;
    cout << "PSNR : " << psnr << " dB" << endl;
    cout << "SSIM : " << ssim << " (1.0 = identik)" << endl;

    showBeforeAfter(original, compressed, "Fidelity Kriteria");
}

// =============================================
// Menu 26: Pengenalan Feature (ORB)
// =============================================
Mat pengenalanFeature(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Ptr<ORB> orb = ORB::create(500);
    vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->detectAndCompute(gray, noArray(), keypoints, descriptors);

    Mat result = img.clone();
    drawKeypoints(img, keypoints, result, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cout << "\n=== Pengenalan Feature (ORB) ===" << endl;
    cout << "Jumlah keypoint terdeteksi: " << keypoints.size() << endl;
    cout << "Ukuran descriptor         : " << descriptors.rows << " x " << descriptors.cols << endl;

    return result;
}

// =============================================
// Menu 27: Pattern Recognition
// =============================================
Mat patternRecognition(const Mat& img) {
    CascadeClassifier face_cascade;
    bool loaded = face_cascade.load("haarcascade_frontalface_default.xml");
    if (!loaded) {
        string cascadePath = cv::samples::findFile("haarcascade_frontalface_default.xml", false);
        if (!cascadePath.empty())
            loaded = face_cascade.load(cascadePath);
    }

    Mat result = img.clone();

    if (!loaded) {
        cout << "\n[Pattern Recognition] Haar Cascade tidak ditemukan." << endl;
        cout << "Menggunakan deteksi kontur sebagai alternatif..." << endl;
        Mat gray, edges;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        Canny(gray, edges, 50, 150);
        vector<vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        int count = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) > 500) {
                rectangle(result, boundingRect(contours[i]), Scalar(0, 255, 0), 2);
                count++;
            }
        }
        cout << "Pola/objek terdeteksi: " << count << endl;
        return result;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
    for (size_t i = 0; i < faces.size(); i++) {
        rectangle(result, faces[i], Scalar(0, 255, 0), 2);
        putText(result, "Wajah " + to_string(i + 1), Point(faces[i].x, faces[i].y - 5),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
    }
    cout << "\n=== Pattern Recognition ===" << endl;
    cout << "Wajah terdeteksi: " << faces.size() << endl;
    return result;
}

// =============================================
// Menu 28: Rekonstruksi Citra (DFT -> IDFT)
// =============================================
Mat rekonstruksiCitra(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);

    int m = getOptimalDFTSize(gray.rows);
    int n = getOptimalDFTSize(gray.cols);
    Mat padded;
    copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { padded, Mat::zeros(padded.size(), CV_32F) };
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);

    split(complexImg, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magImg = planes[0];
    magImg += Scalar::all(1);
    log(magImg, magImg);
    normalize(magImg, magImg, 0, 255, NORM_MINMAX);
    magImg.convertTo(magImg, CV_8U);

    Mat reconstructed;
    idft(complexImg, reconstructed, DFT_REAL_OUTPUT | DFT_SCALE);
    normalize(reconstructed, reconstructed, 0, 255, NORM_MINMAX);
    reconstructed.convertTo(reconstructed, CV_8U);
    reconstructed = reconstructed(Rect(0, 0, gray.cols, gray.rows));

    cout << "\n=== Rekonstruksi Citra (DFT -> IDFT) ===" << endl;
    cout << "Ukuran asli    : " << img.cols << " x " << img.rows << endl;
    cout << "Ukuran padding : " << n << " x " << m << endl;

    imshow("Spectrum Frekuensi", magImg);

    return reconstructed;
}
#include "UAS_PCD_25.h"
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    string path = "C:\\Users\\vsphi\\matkul\\smt 6\\Pengolahan Citra Digital\\UAS_PCD_25_PHILIP ZEFANYA VICTORY S\\Citra\\gambar1.jpg";
    Mat img = imread(path);
    if (img.empty()) {
        cout << "Gagal membuka gambar." << endl;
        return -1;
    }

    int pilihan;
    do {
        cout << "\n=== MENU PCD ===" << endl;
        cout << "1. Tampilkan Citra\n2. Akses Data Citra\n3. Grayscale\n4. Negative\n5. Brightness\n";
        cout << "6. Histogram\n7. Equalize Hist (Gray)\n8. Equalize Hist (Color)\n9. Histogram Matching\n10. Median Blur\n";
        cout << "11. Bilateral Blur\n12. Sharpening\n13. Laplace\n14. Sobel + Canny\n15. Prewitt\n";
        cout << "16. Threshold\n17. Adaptive Threshold\n18. Region Growing\n19. Split & Merge\n20. Clustering\n";
        cout << "21. Dilate & Erode\n22. Open & Close\n23. Hit or Miss\n24. Kompresi\n";
        cout << "25. Fidelity Kriteria\n26. Pengenalan Feature\n27. Pattern Recognition\n28. Rekonstruksi Citra\n29. Exit\n";
        cout << "Pilihan Anda: ";
        cin >> pilihan;

        Mat hasil;
        switch (pilihan) {
        case 1:
            tampilkanCitra(img);
            break;
        case 2:
            aksesDataCitra(img);
            break;
        case 3:
            hasil = ubahKeGrayscale(img);
            showBeforeAfter(img, hasil, "Grayscale");
            break;
        case 4:
            hasil = ubahKeNegative(img);
            showBeforeAfter(img, hasil, "Negative");
            break;
        case 5: {
            int val;
            cout << "Tambah brightness: ";
            cin >> val;
            hasil = tambahBrightness(img, val);
            showBeforeAfter(img, hasil, "Brightness");
            break;
        }
        case 6:
            tampilkanHistogram(img);
            tampilkanCitra(img);
            break;
        case 7:
            hasil = equalizeHistGray(img);
            showBeforeAfter(img, hasil, "Equalize Hist Gray");
            break;
        case 8:
            hasil = equalizeHistColor(img);
            showBeforeAfter(img, hasil, "Equalize Hist Color");
            break;
        case 9: {
            string inputPath = "C:\\Users\\vsphi\\matkul\\smt 6\\Pengolahan Citra Digital\\UAS_PCD_25_PHILIP ZEFANYA VICTORY S\\Citra\\gambar1.jpg";
            string referencePath = "C:\\Users\\vsphi\\matkul\\smt 6\\Pengolahan Citra Digital\\UAS_PCD_25_PHILIP ZEFANYA VICTORY S\\Citra\\gambar3.png";
            histogramMatching(inputPath, referencePath);
            break;
        }
        case 10:
            hasil = applyMedianBlur(img);
            showBeforeAfter(img, hasil, "Median Blur");
            break;
        case 11:
            hasil = applyBilateralBlur(img);
            showBeforeAfter(img, hasil, "Bilateral Blur");
            break;
        case 12:
            hasil = applySharpening(img);
            showBeforeAfter(img, hasil, "Sharpening");
            break;
        case 13:
            hasil = applyLaplace(img);
            showBeforeAfter(img, hasil, "Laplace");
            break;
        case 14:
            hasil = applySobelCanny(img);
            showBeforeAfter(img, hasil, "Sobel + Canny");
            break;
        case 15:
            hasil = applyPrewitt(img);
            showBeforeAfter(img, hasil, "Prewitt");
            break;
        case 16:
            hasil = applyThreshold(img);
            showBeforeAfter(img, hasil, "Threshold");
            break;
        case 17:
            hasil = applyAdaptiveThreshold(img);
            showBeforeAfter(img, hasil, "Adaptive Threshold");
            break;
        case 18:
            hasil = regionGrowing(img, Point(100, 100));
            showBeforeAfter(img, hasil, "Region Growing");
            break;
        case 19:
            hasil = splitAndMerge(img);
            showBeforeAfter(img, hasil, "Split & Merge");
            break;
        case 20:
            hasil = clusteringSegmentation(img);
            showBeforeAfter(img, hasil, "Clustering");
            break;
        case 21:
            hasil = applyDilateErode(img);
            showBeforeAfter(img, hasil, "Dilate & Erode");
            break;
        case 22:
            hasil = applyOpenClose(img);
            showBeforeAfter(img, hasil, "Open & Close");
            break;
        case 23:
            hasil = applyHitOrMiss(img);
            showBeforeAfter(img, hasil, "Hit or Miss");
            break;
        case 24:
            hasil = compressImage(img);
            showBeforeAfter(img, hasil, "Kompresi");
            break;
        case 25: {
            Mat compressed = compressImage(img);
            fidelityKriteria(img, compressed);
            break;
        }
        case 26:
            hasil = pengenalanFeature(img);
            showBeforeAfter(img, hasil, "Pengenalan Feature");
            break;
        case 27:
            hasil = patternRecognition(img);
            showBeforeAfter(img, hasil, "Pattern Recognition");
            break;
        case 28:
            hasil = rekonstruksiCitra(img);
            showBeforeAfter(img, hasil, "Rekonstruksi Citra");
            break;
        case 29:
            cout << "Keluar...\n";
            break;
        default:
            cout << "Pilihan tidak valid.\n";
            break;
        }

        if (pilihan != 29) waitKey(0);
        destroyAllWindows();

    } while (pilihan != 29);

    return 0;
}
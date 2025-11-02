#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

int main() {
    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DeeplabV3Plus");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        const wchar_t* model_path = L"deeplabv3_plus_v2_LaPa.onnx";
        Ort::Session session(env, model_path, session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        const int input_size = 512;

        // Load all image paths from folder
        std::string folder_path = "images"; //face-tests
        std::vector<std::string> image_paths;
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext == ".jpg" || ext == ".png" || ext == ".jpeg")
                    image_paths.push_back(entry.path().string());
            }
        }

        if (image_paths.empty()) {
            std::cerr << "❌ No images found in folder: " << folder_path << std::endl;
            return -1;
        }

        // Shuffle and pick 8 random images
        std::shuffle(image_paths.begin(), image_paths.end(), std::mt19937(std::random_device{}()));
        if (image_paths.size() > 8)
            image_paths.resize(8);

        // CATEGORY_COLORS
        std::vector<cv::Vec3b> CATEGORY_COLORS = {
            {0, 0, 0},
            {0, 153, 255},
            {102, 255, 153},
            {0, 204, 153},
            {255, 255, 102},
            {255, 255, 204},
            {255, 153, 0},
            {255, 102, 255},
            {102, 0, 51},
            {255, 204, 255},
            {102, 0, 255}
        };

        // Prepare I/O names
        Ort::AllocatedStringPtr input_name_alloc = session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr output_name_alloc = session.GetOutputNameAllocated(0, allocator);
        const char* input_names[] = { input_name_alloc.get() };
        const char* output_names[] = { output_name_alloc.get() };

        std::vector<cv::Mat> results;

        for (const auto& image_path : image_paths) {
            cv::Mat img_bgr = cv::imread(image_path);
            if (img_bgr.empty()) {
                std::cerr << "⚠️ Skipping invalid image: " << image_path << std::endl;
                continue;
            }

            // Preprocess
            cv::Mat img_rgb;
            cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
            cv::resize(img_rgb, img_rgb, cv::Size(input_size, input_size));
            img_rgb.convertTo(img_rgb, CV_32F);

            std::vector<float> input_tensor_values;
            input_tensor_values.reserve(input_size * input_size * 3);
            for (int y = 0; y < input_size; ++y) {
                for (int x = 0; x < input_size; ++x) {
                    cv::Vec3f px = img_rgb.at<cv::Vec3f>(y, x);
                    input_tensor_values.push_back(px[0]);
                    input_tensor_values.push_back(px[1]);
                    input_tensor_values.push_back(px[2]);
                }
            }

            std::vector<int64_t> input_dims = { 1, input_size, input_size, 3 };
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem_info, input_tensor_values.data(), input_tensor_values.size(),
                input_dims.data(), input_dims.size()
            );

            // Run inference
            auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                input_names, &input_tensor, 1, output_names, 1);

            auto& out_tensor = output_tensors.front();
            float* out_data = out_tensor.GetTensorMutableData<float>();
            auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
            int h = (int)out_shape[1];
            int w = (int)out_shape[2];
            int num_classes = (int)out_shape[3];

            // Argmax per pixel
            cv::Mat mask(h, w, CV_8UC1);
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j) {
                    float max_val = out_data[(i * w + j) * num_classes];
                    int max_idx = 0;
                    for (int c = 1; c < num_classes; ++c) {
                        float val = out_data[(i * w + j) * num_classes + c];
                        if (val > max_val) { max_val = val; max_idx = c; }
                    }
                    mask.at<uchar>(i, j) = (uchar)max_idx;
                }
            }

            // Convert mask to color
            cv::Mat color_mask(h, w, CV_8UC3);
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j) {
                    int cls = mask.at<uchar>(i, j);
                    cv::Vec3b color = CATEGORY_COLORS[cls % CATEGORY_COLORS.size()];
                    color_mask.at<cv::Vec3b>(i, j) = color;
                }
            }

            // Blend overlay
            cv::Mat img_resized;
            cv::resize(img_bgr, img_resized, cv::Size(w, h));
            cv::Mat blended;
            cv::addWeighted(img_resized, 0.5, color_mask, 0.5, 0, blended);
            results.push_back(blended);
        }

        if (results.empty()) {
            std::cerr << "❌ No valid results to display." << std::endl;
            return -1;
        }

        // Create grid (4 cols × 2 rows)
        int cols = 4, rows = 2;
        int thumb_w = 256, thumb_h = 256;
        cv::Mat grid(rows * thumb_h, cols * thumb_w, CV_8UC3, cv::Scalar(0, 0, 0));

        for (int idx = 0; idx < results.size() && idx < rows * cols; ++idx) {
            cv::Mat thumb;
            cv::resize(results[idx], thumb, cv::Size(thumb_w, thumb_h));
            int r = idx / cols;
            int c = idx % cols;
            thumb.copyTo(grid(cv::Rect(c * thumb_w, r * thumb_h, thumb_w, thumb_h)));
        }

        cv::imshow("Segmentation Grid", grid);
        cv::waitKey(0);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "ReadMnistData.h"

namespace mnist {
// Class to represent a Multi-Layer Perceptron (MLP) model
class MnistMlp {
    cv::Mat layer_sizes;
    cv::Ptr<cv::ml::ANN_MLP> mlp;
    cv::TermCriteria term_criteria;
    std::vector<cv::Mat> imagesData;  // Store your images
    std::vector<int> labelsData;      // Corresponding labels

public:
    // Specify parameters during construction
    MnistMlp(int input_layer_size, int hidden_layer_size, int output_layer_size,
              double learning_rate = 0.001, double momentum = 0.1) {
        layer_sizes = (cv::Mat_<int>(3, 1) << input_layer_size, hidden_layer_size, output_layer_size);
        mlp = cv::Ptr<cv::ml::ANN_MLP>(cv::ml::ANN_MLP::create());
        mlp->setLayerSizes(layer_sizes);
        mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
        mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, learning_rate, momentum);
        term_criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, 0.001);
    }

    void prepdata(const ReadMnistData& data)
    {
        for(int imgCnt=0; imgCnt<(int)data.getImages().size(); imgCnt++)
        {
            int rowCounter = 0;
            int colCounter = 0;

            cv::Mat tempImg = cv::Mat::zeros(cv::Size(28,28),CV_8UC1);
            for (int i = 0; i < (int)data.getImages()[imgCnt].size(); i++) {

            tempImg.at<uchar>(cv::Point(colCounter++,rowCounter)) = (int)data.getImages()[imgCnt][i];
            if ((i) % 28 == 0) {
                rowCounter++;
                colCounter= 0;
                if(i == 756)
                    break;
            }
            }
            //std::cout<<(int)data.getLabels()[imgCnt][0]<<std::endl;
            imagesData.push_back(tempImg);
            labelsData.push_back((int)data.getLabels()[imgCnt][0]);
        }

    }

    // Load the model from a serialized XML file
    void load(const std::string& filename) {
        mlp = cv::ml::ANN_MLP::load(filename);
    }

    // Save the trained model to an XML file
    void save(const std::string& filename) const { 
        mlp->save(filename);
    }

    // Train the MLP model on the provided data
    void train(const ReadMnistData& data) {
        int num_samples = imagesData.size();
        // Prepare training data as in original code
        cv::Mat training_data(num_samples, data.getImages()[0].size(), CV_32F);
        cv::Mat label_data(num_samples, 10, CV_32F);

        for (int i = 0; i < num_samples; ++i) {
            cv::Mat image = imagesData[i].reshape(1, 1);
            image.convertTo(training_data.row(i), CV_32F);

            cv::Mat label = cv::Mat::zeros(1, 10, CV_32F);
            label.at<float>(0, labelsData[i]) = 1.0f; // Assuming single label

            label.copyTo(label_data.row(i));
        }

        mlp->setTermCriteria(term_criteria);
        mlp->train(training_data, cv::ml::ROW_SAMPLE, label_data); 
    }
};
}

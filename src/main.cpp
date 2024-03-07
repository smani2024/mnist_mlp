#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "ReadMnistData.h"
#include "MnistMLP.h"

// Include the MnistData and MnistMlp classes from the "mnist" namespace
using namespace mnist;

int main() {
    std::string images_filename = "/media/work/MNIST_dataset/train-images-idx3-ubyte/train-images.idx3-ubyte";
    std::string labels_filename = "/media/work/MNIST_dataset/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
    std::vector<cv::Mat> imagesData;  // Store your images
    std::vector<int> labelsData;      // Corresponding labels

    try {
        // Load the MNIST dataset
        ReadMnistData data(images_filename, labels_filename);

        //std::cout<<(int)data.getImages().size()<<std::endl;
        //std::cout<<(int)data.getLabels().size()<<std::endl;



        // Create an MLP with appropriate layer sizes
        int input_layer_size = data.getImages()[0].size();
        int hidden_layer_size = 100;
        int output_layer_size = 10;
        MnistMlp model(input_layer_size, hidden_layer_size, output_layer_size);

        model.prepdata(data);

        // Train the model
        model.train(data);

        // Save the trained model
        model.save("/media/work/train/mnist_mlp/MNIST_train_model.xml");

        std::cout << "Model trained and saved successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


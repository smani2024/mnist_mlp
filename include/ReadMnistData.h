#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

namespace mnist {

class ReadMnistData {
public:
    // Load the images and labels from the specified files
    ReadMnistData(const std::string& images_filename, const std::string& labels_filename) {
        imagesFile = readIDX3UByteFile(images_filename);
        labelsFile = readLabelFile(labels_filename);

        if (imagesFile.size() != labelsFile.size()) {
            throw std::runtime_error("Number of images and labels must be equal!");
        }
    }

    // Getters for images and labels
    const std::vector<std::vector<unsigned char>>& getImages() const { return imagesFile; }
    const std::vector<std::vector<unsigned char>>& getLabels() const { return labelsFile; }

private:
    // Helper functions to read IDX3-UBYTE files - need to convert as .a
    std::vector<std::vector<unsigned char>> readIDX3UByteFile(const std::string& filename);

    std::vector<std::vector<unsigned char>> readLabelFile(const std::string& filename);

    std::vector<std::vector<unsigned char>> imagesFile;
    std::vector<std::vector<unsigned char>> labelsFile;
};

}

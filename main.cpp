// main.cpp
#include <Eigen/Dense>
#include <chrono>
#include <fdeep/fdeep.hpp>
#include <yarp/os/Property.h>
#include <yarp/os/ResourceFinder.h>

#include <iostream>

Eigen::MatrixXi skinMapping;
bool parseMatrix(const yarp::os::Searchable& rf,
                 const std::string& key,
                 Eigen::Ref<Eigen::MatrixXi> matrix)
{
    yarp::os::Value ini = rf.find(key);
    if (ini.isNull() || !ini.isList())
    {
        return false;
    }

    yarp::os::Bottle* outerList = ini.asList();

    for (int row = 0; row < outerList->size(); ++row)
    {
        yarp::os::Value& innerValue = outerList->get(row);
        if (innerValue.isNull() || !innerValue.isList())
        {
            return false;
        }
        yarp::os::Bottle* innerList = innerValue.asList();

        for (int column = 0; column < innerList->size(); ++column)
        {
            matrix(row, column) = int(innerList->get(column).asFloat64());
        }
    }

    return true;
}

Eigen::MatrixXf getSkinMarix(const Eigen::VectorXf& skinRaw)
{
    Eigen::MatrixXf matrix = Eigen::MatrixXf::Zero(9, 11);

    for (int i = 0; i < skinMapping.rows(); i++)
    {
        int row = skinMapping(i, 1);
        int col = skinMapping(i, 2);
        int indexRawData = skinMapping(i, 0);
        matrix(row, col) = skinRaw(indexRawData);
    }

    return matrix;
}

int main()
{
    const std::string pathToModel = "../model/model_images_v0.json";
    auto network = std::make_unique<fdeep::model>(fdeep::load_model(pathToModel));

    yarp::os::ResourceFinder& rf = yarp::os::ResourceFinder::getResourceFinderSingleton();
    rf.setDefaultConfigFile("config.ini");
    std::vector<char*> argv;
    argv.push_back(nullptr);
    rf.configure(argv.size() - 1, argv.data());

    skinMapping.resize(48, 3);
    parseMatrix(rf, "palm_skin_mapping", skinMapping);

    Eigen::VectorXf skinRaw = Eigen::VectorXf::Zero(48);

    // different skin data
    // skinRaw << 0.029872, 0.003748, 0.002575, 0.00254, 0.015936, 0.106104, 0.003792, 0.002746,
    // 0.003432, 0.006272, 0.008587, 0.0, 0.0, 0.035089, 0.121358, 0.009344, 0.46876, 0.485073,
    // 0.006019, 0.243866, 0.894014, 0.077499, 0.0, 0.0, 0.001094, 0.0, 0.002774, 0.0, 0.000533,
    // 0.000473, 0.0, 0.000939, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // 0.0, 0.0, 0.0;

    // skinRaw << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004497, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004923,
    // 0.0, 0.075726, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    skinRaw << 0.016324, 0.011052, 0.0, 0.0, 0.032045, 0.00948, 0.0, 0.041052, 0.356333, 0.231177,
        0.016335, 0.008576, 0.004727, 0.007105, 0.000235, 0.002808, 0.0, 0.0, 0.005235, 0.039933,
        0.0, 0.001473, 0.010042, 0.0, 0.0, 0.0, 0.0, 0.000712, 0.001597, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    Eigen::MatrixXf matrix = getSkinMarix(skinRaw);
    std::cerr << matrix << std::endl;

    // transform eigen matrix in fdeep tensor
    const int tensor_channels = 1;
    const int tensor_rows = matrix.rows();
    const int tensor_cols = matrix.cols();
    fdeep::tensor_shape tensor_shape(tensor_rows, tensor_cols, tensor_channels);
    fdeep::tensor t(tensor_shape, 0.0f);

    // copy the values into tensor
    for (int y = 0; y < tensor_rows; ++y)
    {
        for (int x = 0; x < tensor_cols; ++x)
        {
            for (int c = 0; c < tensor_channels; ++c)
            {
                t.set(fdeep::tensor_pos(y, x, c), matrix(y, x));
            }
        }
    }

    while (true)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        const auto result = network->predict_single_output({t});

        std::string rockType = result < 0.5 ? "plain" : "rough";
        std::cerr << "Rock: " << rockType << ". Neural Network outcome: " << result << "."
                  << std::endl;

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time difference = "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << "[µs]" << std::endl;

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(1000ms);
    }
    return 0;
}

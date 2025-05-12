#include "tensor.h"
#include "OpKernel/Conv/NormalConv.h"
#include "zcom_tick.h"
#include <fstream>

#define FILE_W 1920
#define FILE_H 1080
#define FILE_C 3
//#define FILE_PATH_IN "E:\\files\\视频文件\\RGB\\dynamic_rgb_1920x1080_1F.bin"
//#define FILE_PATH_OUT "E://files//视频文件//RGB//dynamic_hri_1920x1080_1F.bin"
#define FILE_PATH_IN "./rgb_1920x1080_1F.bin"
#define FILE_PATH_OUT "./oneplane_1920x1080_1F.bin"

int ReadOnePlaneImg(const std::string& filepath, uint32_t filewidth, uint32_t fileheight, uint32_t planenum, std::vector<int>& outdata)
{
    uint32_t imglen = filewidth * fileheight * planenum;
    std::vector<int> imgdata;
    imgdata.resize(imglen);

    std::fstream filehd(filepath.c_str(), std::ios::in | std::ios::binary);
    if (filehd.is_open())
    {
        filehd.read((char*)imgdata.data(), imglen);
        filehd.close();
    }
    else
    {
        return -1;
    }

    uint32_t planesize = filewidth * fileheight;
    for (int r=0; r<planesize; r++)
    {
        outdata.push_back(imgdata[r * planenum]);
    }

    return 0;
}

int WriteOnePlaneImg(const std::string& filepath, const std::vector<int>& imgdata)
{
    std::fstream filehd(filepath, std::ios::app | std::ios::binary);
    if (filehd.is_open())
    {
        filehd.write((char*)imgdata.data(), imgdata.size());
    }
    else
    {
        return -1;
    }
    filehd.close();

    return 0;
}

// 水平Sobel边缘检测核
void TestHorizontalEdge() {
    using TensorType = zinfer::Tensor<float>;

    // 初始化5x5x1输入（中间水平线）
    TensorType input(5, 5, 1);
    input.data().fill(0);
    input.data().row(2).fill(1); // 中间行置1
    input.Show();

    // 水平Sobel核 [-1 0 1; -2 0 2; -1 0 1]， 检测垂直边缘
    // 垂直Sobel核 [-1 -2 -1; 0 0 0; 1 2 1]， 检测水平边缘
    TensorType sobel_x(3, 3, 1);
    sobel_x.Fill({ -1, 0, 1, -2, 0, 2, -1, 0, 1 }, false);  // 水平Sobel核 + 列优先存储 = 垂直Sobel核
    sobel_x.Show();
    //sobel_x.data() << -1 << 0 << 1 << arma::endr
    //    << -2 << 0 << 2 << arma::endr
    //    << -1 << 0 << 1;

    // 配置卷积层
    zinfer::Convolution_Normal<float> conv(
        1,  // padding=1保证输出同尺寸
        1,  // stride=1
        false // 无偏置
    );

    std::vector<TensorType> filters = { sobel_x };
    conv.SetFilters(filters);

    std::vector<TensorType> inputs = { input };
    conv.SetInputs(inputs);

    // 执行卷积
    std::vector<TensorType> outputs;
    conv.DoConv(outputs);

    // 验证输出特征
    const auto& out = outputs[0].data();
    std::cout << "Horizontal Edge Detection Result:\n" << out.slice(0) << std::endl;

    //// 预期中间行响应最强
    //assert(out(2, 2, 0) == 4.0f); // 中心响应
    //assert(arma::accu(out) > 15.0f); // 总响应强度
}

// 水平Sobel边缘检测核
void TestHorizontalEdgeToFile()
{
    std::vector<int> imgdata;
    int res = ReadOnePlaneImg(FILE_PATH_IN, FILE_W, FILE_H, FILE_C, imgdata);
    if (res != 0)
    {
        return;
    }
    //WriteOnePlaneImg("./temp_in.bin", imgdata);

    using TensorType = zinfer::Tensor<int>;

    // 初始化5x5x1输入（中间水平线）
    TensorType input(FILE_W, FILE_H, 1);
    input.Fill(imgdata);
    //input.Show();

    // 水平Sobel核 [-1 0 1; -2 0 2; -1 0 1]， 检测垂直边缘
    // 垂直Sobel核 [-1 -2 -1; 0 0 0; 1 2 1]， 检测水平边缘
    TensorType sobel_x(3, 3, 1);
    sobel_x.Fill({ -1, 0, 1, -2, 0, 2, -1, 0, 1 }, false);  // 水平Sobel核 + 列优先存储 = 垂直Sobel核
    //sobel_x.Show();

    // 配置卷积层
    zinfer::Convolution_Normal<int> conv(
        1,  // padding=1保证输出同尺寸
        1,  // stride=1
        false // 无偏置
    );

    std::vector<TensorType> filters = { sobel_x };
    conv.SetFilters(filters);

    std::vector<TensorType> inputs = { input };
    conv.SetInputs(inputs);

    auto tick_in = zmodule::zcom::tick_time::now_mills();
    // 执行卷积
    std::vector<TensorType> outputs;
    conv.DoConv(outputs);
    auto tick_out = zmodule::zcom::tick_time::now_mills();
    std::cout << "doConv cost " << (tick_out - tick_in) << "ms" << std::endl;

    // 验证输出特征
    const auto& out = outputs[0].data();
    //std::cout << "Horizontal Edge Detection Result:\n" << out.slice(0) << std::endl;

    const auto& outdata = outputs[0].Values();
    WriteOnePlaneImg(FILE_PATH_OUT, outdata);

}


int main()
{
    //zinfer::Tensor<float> ftensor(1);
    //ftensor.Show();
    //
    //ftensor.Ones();
    //ftensor.Show();
    //
    //zinfer::Tensor<float> ftensor1(10, 10, 10);
    //ftensor1.Show();
    //
    //std::vector<float> values;
    //values.resize(1000);
    //for (int r = 0; r < 1000; r++)
    //{
    //    values[r] = r;
    //}
    ////ftensor1.Fill(values, false);
    //ftensor1.Fill(values, true);
    ////ftensor1.Show();
    ////
    ////zinfer::TensorShape newshape(10, 20, 5);
    //////ftensor1.Reshape(newshape, true);
    ////ftensor1.Reshape(newshape, false);
    ////ftensor1.Show();
    //
    ////zinfer::Tensor<float> ftensor2(ftensor1);  /* 拷贝 */
    ////ftensor2.Show();
    ////ftensor2.Ones();
    ////ftensor1.Show();
    //
    ////zinfer::Tensor<float> ftensor2(std::move(ftensor1));  /* 转移 */
    ////ftensor2.Show();
    ////ftensor2.Ones();
    ////ftensor1.Show();
    //
    ////auto& date_channel = ftensor1.data_slice(2);
    ////INFO_TRACE(date_channel);
    //
    //
    //zinfer::Tensor<uint32_t> uitensor(2, 2, 1);
    //uitensor.Show();
    //
    //std::vector<uint32_t> pad_border;
    //pad_border.reserve(4);
    //pad_border.push_back(1);
    //pad_border.push_back(1);
    //pad_border.push_back(1);
    //pad_border.push_back(1);
    //
    //uitensor.Padding(pad_border, 1);
    //uitensor.Show();
    //
    //
    //INFO_TRACE(uitensor.at(1, 2, 2));
    //
    //std::string datas_str;
    //for (int r = 0; r < uitensor.size(); r++)
    //{
    //    datas_str += std::to_string(uitensor.index(r));
    //}
    //INFO_TRACE(datas_str);
    //
    //
    //uitensor.Transform([&](uint32_t item)->uint32_t {return item * 5; });
    //uitensor.Show();
    //
    //
    //uitensor.Flatten();
    //uitensor.Show();


    //zinfer::Tensor<float> tensor1(3, 1, 3);
    //tensor1.Rand();
    //tensor1.Transform([](uint32_t item)->uint32_t {return item / fabs(item); });
    //tensor1.Show();
    //
    //
    //zinfer::Tensor<float> tensor2(3, 2, 1);
    //tensor2.Rand();
    //tensor2.Transform([](uint32_t item)->uint32_t {return item / fabs(item); });
    //tensor2.Show();
    //
    ////auto tensor3 = tensor1 + tensor2;
    ////auto tensor3 = tensor1 - tensor2;
    ////auto tensor3 = tensor1 % tensor2;
    //auto tensor3 = tensor1 / tensor2;
    //tensor3.Show();


    //zinfer::Tensor<float> tensor1(3, 1, 3);
    //tensor1.Rand();
    //tensor1.Transform([](uint32_t item)->uint32_t {return item / fabs(item); });
    //tensor1.Show();
    //
    //
    //zinfer::Tensor<float> tensor2(1, 2, 3);
    //tensor2.Rand();
    //tensor2.Transform([](uint32_t item)->uint32_t {return item / fabs(item); });
    //tensor2.Show();
    //
    //auto tensor3 = tensor1 * tensor2;
    //tensor3.Show();


    //zinfer::Tensor<float> tensor1(300, 10, 300);
    //tensor1.Rand();
    ////tensor1.Transform([](uint32_t item)->uint32_t {return item / fabs(item); });
    ////tensor1.Show();
    //
    //
    //zinfer::Tensor<float> tensor2(10, 200, 300);
    //tensor2.Rand();
    ////tensor2.Transform([](uint32_t item)->uint32_t {return item / fabs(item); });
    ////tensor2.Show();
    //
    //
    //auto intick = zmodule::zcom::tick_time::now_nanos();
    //auto tensor3 = tensor1 * tensor2;
    //auto outtick = zmodule::zcom::tick_time::now_nanos();
    //INFO_TRACE("time_cost=" << (outtick - intick));
    ////tensor3.Show();

    //zinfer::Tensor<float> tensor1(3, 2, 1);
    //tensor1.Rand();
    //tensor1.Transform([](uint32_t item)->uint32_t {return item / fabs(item); });
    //tensor1.Show();
    //
    //
    //zinfer::Tensor<float> tensor2(3, 2, 1);
    //tensor2.Rand();
    //tensor2.Transform([](uint32_t item)->uint32_t {return item / fabs(item); });
    //tensor2.Show();
    //
    ////auto tensor3 = tensor1 + tensor2;
    ////auto tensor3 = tensor1 - tensor2;
    ////auto tensor3 = tensor1 % tensor2;
    //auto tensor3 = tensor1 / tensor2;
    //tensor3.Show();
    //
    ////zinfer::Convolution_Normal<float> nConv(1, 2, true);

    //TestHorizontalEdge();

    TestHorizontalEdgeToFile();


    //zinfer::Tensor<int> input(4, 3, 2);
    //input.Rand();
    //input.Show();
    //
    //std::string output = "";
    //for (int r = 0; r < input.size(); r++)
    //{
    //    output += std::to_string((int)input.rawptr()[r]) + " ";
    //}
    //std::cout << output << std::endl;

    system("pause");
    return 0;
}

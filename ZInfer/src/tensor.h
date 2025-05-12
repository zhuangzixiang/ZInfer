#ifndef __ZINFER_TENSOR_H__
#define __ZINFER_TENSOR_H__

//#define ARMA_USE_BLAS
//#define ARMA_USE_LAPACK

#include <armadillo>
#include <memory>
#include <vector>

#include "zcom_trace.h"

namespace zinfer
{
class TensorShape
{
public:
    TensorShape() : rows_(0), cols_(0), chns_(0) {}
    TensorShape(uint32_t rows, uint32_t cols, uint32_t chns) : rows_(rows), cols_(cols), chns_(chns) {}

    void setRows(uint32_t rows) { rows_ = rows; }
    void setCols(uint32_t cols) { cols_ = cols; }
    void setChns(uint32_t chns) { chns_ = chns; }
    uint32_t Rows() const { return rows_; }
    uint32_t Cols() const { return cols_; }
    uint32_t Chns() const { return chns_; }
    uint32_t Size() const { return rows_ * cols_ * chns_; }
    uint32_t Dims() const { return 3; }

    bool operator==(const TensorShape& other) const noexcept
    {
        return rows_ == other.rows_ && cols_ == other.cols_ && chns_ == other.chns_;
    }

    bool operator!=(const TensorShape& other) const noexcept
    {
        return !(*this == other);
    }

    bool operator<(const TensorShape& other) const noexcept
    {
        //if (rows_ != other.rows_) return rows_ < other.rows_;
        //if (cols_ != other.cols_) return cols_ < other.cols_;
        //if (chns_ != other.chns_) return chns_ < other.chns_;
        //return chns_ < other.chns_;
        if (rows_ < other.rows_ && cols_ < other.cols_ && chns_ < other.chns_)
        {
            return true;
        }
        else
        {
            return Size() < other.Size();
        }
    }

    bool operator>(const TensorShape& other) const noexcept
    {
        return other < *this;
    }

    bool operator<=(const TensorShape& other) const noexcept
    {
        return !(other < *this);
    }

    bool operator>=(const TensorShape& other) const noexcept
    {
        return !(*this < other);
    }

    TensorShape(const TensorShape& other) noexcept : rows_(other.rows_),
        cols_(other.cols_), chns_(other.chns_)
    {}

    TensorShape& operator=(const TensorShape& other) noexcept
    {
        if (this != &other)
        {
            rows_ = other.rows_;
            cols_ = other.cols_;
            chns_ = other.chns_;
        }
        return *this;
    }

    TensorShape(const TensorShape&& other) noexcept : rows_(other.rows_),
        cols_(other.cols_), chns_(other.chns_)
    {}

    /*** 移动赋值运算符 ***/
    TensorShape& operator=(const TensorShape&& other) noexcept
    {
        if (this != &other)
        {
            rows_ = other.rows_;
            cols_ = other.cols_;
            chns_ = other.chns_;
        }
        return *this;
    }

private:
    /*** 先有行，再有列，然后才会有channel ***/
    uint32_t rows_ = 0;
    uint32_t cols_ = 1;
    uint32_t chns_ = 1;
};

/**************************************************/
/*  类型映射系统                                  */
/**************************************************/
template <typename T>
struct ArmaType
{
    using cubetype = arma::Cube<T>;
    using mattype = arma::Mat<T>;
    using rowtype = arma::Row<T>;
};

template <>
struct ArmaType<float>
{
    using cubetype = arma::fcube;
    using mattype = arma::fmat;
    using rowtype = arma::frowvec;
};

template <>
struct ArmaType<double>
{
    using cubetype = arma::cube;
    using mattype = arma::dmat;
    using rowtype = arma::rowvec;
};

template <>
struct ArmaType<int>
{
    using cubetype = arma::s32_cube; 
    using mattype = arma::s32_mat;
    using rowtype = arma::s32_rowvec;
};

template <>
struct ArmaType<uint32_t>
{
    using cubetype = arma::Cube<uint32_t>; 
    using mattype = arma::Mat<uint32_t>;
    using rowtype = arma::u32_rowvec;
};


/**************************************************/
/***  Tensor 类声明                              ***/
/**************************************************/
template <typename T>
class Tensor
{
public:
    using ArmaCubeType = typename ArmaType<T>::cubetype;
    using ArmaMatType = typename ArmaType<T>::mattype;
    explicit Tensor() : data_(), rawdata_size_() {}
    explicit Tensor(ArmaCubeType&& data) : data_(std::move(data)), rawdata_size_(data_.n_rows, data_.n_cols, data_.n_slices) {}
	virtual ~Tensor() = default;

    /************************************************************************/
    /* 创建                                                               ***/
    /* @param rows 张量的行数                                              ***/
    /* @param cols 张量的列数                                              ***/
    /* @param channels 张量的通道数                                        ***/
    /************************************************************************/
    explicit Tensor(uint32_t rows, uint32_t cols=1, uint32_t channels=1) : data_(rows, cols, channels), rawdata_size_(rows, cols, channels)
    {}
    explicit Tensor(const TensorShape& shape) : data_(shape.Rows(), shape.Cols(), shape.Chns()), rawdata_size_(shape.Rows(), shape.Cols(), shape.Chns())
    {}

    /***
    ** 拷贝
    ** @param shapes 张量的维度
    ***/
    Tensor(const Tensor& other) : data_(other.data_), rawdata_size_(other.rawdata_size_), data_rowmajor_(other.data_rowmajor_) {}
    Tensor& operator=(const Tensor& other)
	{
        if (this != &other)
		{
            data_ = other.data_;
			rawdata_size_ = other.rawdata_size_;
            data_rowmajor_ = other.data_rowmajor_;
        }
        return *this;
    }
    
	/***
    ** 移动
    ** @param shapes 张量的维度
    ***/
    Tensor(Tensor&& other) noexcept : data_(std::move(other.data_)), rawdata_size_(std::move(other.rawdata_size_)), data_rowmajor_(other.data_rowmajor_){}
    Tensor& operator=(Tensor&& other) noexcept
	{
        if (this != &other)
		{
            data_ = std::move(other.data_);
            rawdata_size_ = std::move(other.rawdata_size_);
            data_rowmajor_ = other.data_rowmajor_;
        }
        return *this;
    }
    
    /***
    ** 返回张量的行数
    ** @return 张量的行数
    ***/
    uint32_t rows() const { return data_.n_rows; }
    
    /***
    ** 返回张量的列数
    ** @return 张量的列数
    ***/
    uint32_t cols() const { return data_.n_cols; }
    
    /***
    ** 返回张量的通道数
    ** @return 张量的通道数
    ***/
    uint32_t channels() const { return data_.n_slices; }
    
    /**
     * 返回张量中元素的数量
     * @return 张量的元素数量
     */
    uint32_t size() const { return data_.size(); }
    
    /**
     * 返回张量是否为空
     * @return 张量是否为空
     */
    bool empty() const { return data_.empty(); }
    
    /**
     * 张量的尺寸大小
     * @return 张量的尺寸大小
     */
    TensorShape shapes() const { return TensorShape(data_.n_rows, data_.n_cols, data_.n_slices); }
    
    /***
    ** 设置张量中的具体数据
    ** @param data 数据
    ** @note: Armadillo自动处理维度变化和内存管理
    ***/
    void set_data(const ArmaCubeType& data) { data_ = data; }  /*** 拷贝 ***/
    void set_data(ArmaCubeType&& data) { data_ = std::move(data); }  /*** 移动 ***/
    //void set_data(ArmaCubeType&& data) { data_ = std::forward<ArmaCubeType>(data); }  /*** 完美转发：拷贝/移动 ***/
	
    /***
    *** 返回张量中的数据
    *** @return 张量中的数据
    ***/
    ArmaCubeType& data() { return data_;}
    const ArmaCubeType& data() const { return data_;}
	
    /**
     * 返回张量第channel通道中的数据
     * @param channel 需要返回的通道
     * @return 返回的通道
     */
    ArmaMatType& data_slice(uint32_t channel) { return data_.slice(channel);}
    const ArmaMatType& data_slice(uint32_t channel) const { return data_.slice(channel);}
    
    /**
     * 返回张量中offset位置的元素
     * @param offset 需要访问的位置
     * @return offset位置的元素
     */
    T index(uint32_t offset) const
    { 
        if(offset < data_.size())
		{
			return data_.at(offset);
		}
		return 0;
    }
	
    T& index(uint32_t offset)
    { 
        if(offset < data_.size())
		{
			return data_.at(offset);
		}
        throw std::out_of_range("Tensor::index: offset(" + std::to_string(offset) + ") out of range( " + std::to_string(data_.size()) + ")");
    }
	
    /**
     * 返回特定位置的元素
     * @param channel 通道
     * @param row 行数
     * @param col 列数
     * @return 特定位置的元素
     */
    T at(uint32_t channel, uint32_t row, uint32_t col) const { return data_.at(row, col, channel); }
    T& at(uint32_t channel, uint32_t row, uint32_t col) { return data_.at(row, col, channel); }

    /***
    ** 张量的实际尺寸大小
    ** @return 张量的实际尺寸大小
    ***/
    //const std::vector<uint32_t>& raw_shapes() const;


    /**
     * 填充张量
     * @param pads 填充张量的尺寸
     * @param padding_value 填充张量
     * @note: pytorch中填充模式有几种，比如'constant'、'reflect'、'replicate'、'circular'等。
     *        'constant'是填充固定值，
     *        'reflect'是镜像反射边界，
     *        'replicate'是复制边缘值，
     *        'circular'则是循环填充，常用于周期信号
     * 本函数实现为固定值填充，且目前只对row和col两个维度进行padding
     */
    void Padding(const std::vector<uint32_t>& pads, T padding_value)
    {
        /*** 原始维度 ***/
        const uint32_t rows = this->data_.n_rows;
        const uint32_t cols = this->data_.n_cols;
        const uint32_t channels = this->data_.n_slices;
        
        /*** 填充尺寸 ***/
        const uint32_t pad_rows1 = pads.at(0);  /*** 上方填充行数 ***/
        const uint32_t pad_rows2 = pads.at(1);  /*** 下方填充行数 ***/
        const uint32_t pad_cols1 = pads.at(2);  /*** 左侧填充列数 ***/
        const uint32_t pad_cols2 = pads.at(3);  /*** 右侧填充列数 ***/
		
		/*** 填充后的大小 ***/
        const uint32_t new_rows = rows + pad_rows1 + pad_rows2;
        const uint32_t new_cols = cols + pad_cols1 + pad_cols2;
        
        /*** 创建填充后的cube并初始化 ***/
        ArmaCubeType padded_cube(new_rows, new_cols, channels);
        padded_cube.fill(padding_value);  /*** 先全部按照padding_value填充整个Tensor ***/
		
        /*** 目标区域坐标计算 ***/
        const uint32_t row_start = pad_rows1;
        const uint32_t row_end = row_start + rows - 1;
        const uint32_t col_start = pad_cols1;
        const uint32_t col_end = col_start + cols - 1;
        for (uint32_t r = 0; r < channels; ++r) /*** 逐通道填充 ***/
		{
            /*** 原始数据区域 ***/
            padded_cube.slice(r).submat(row_start, col_start, row_end, col_end) = this->data_.slice(r);
        }
		
        /*** 替换原始数据 ***/
        this->data_ = std::move(padded_cube);
		this->rawdata_size_ = TensorShape(new_rows, new_cols, channels);
    }

    /**
     * 使用value值去初始化向量
     * @param value
     */
    void Fill(T value) { data_.fill(value); }

    /**
     * 使用values中的数据初始化张量
     * @param values 用来初始化张量的数据
     */
    void Fill(const std::vector<T>& values, bool row_major = true)
    {
	    if (row_major)
		{
            const uint32_t rows = this->rows();
            const uint32_t cols = this->cols();
            const uint32_t size_per_channel = rows * cols;
            const uint32_t channels = this->data_.n_slices;
	        
            for (uint32_t r = 0; r < channels; r++)
			{
                auto& channel_data = this->data_.slice(r);
				
				/*** Armadillo 数学库默认的数据存储方式就是列主序,所以创建clos * rows大小，然后转置 ***/
                const arma::Mat<T>& channel_data_t = arma::Mat<T>(values.data() + r * size_per_channel, cols, rows);
                channel_data = channel_data_t.t();
            }
        }
		else
		{
            std::copy(values.begin(), values.end(), this->data_.memptr());
        }
        data_rowmajor_ = row_major;
    }

    /**
     * 返回Tensor内的所有数据
     * @param row_major 是否是行主序列的
     * @return Tensor内的所有数据
     */
    std::vector<T> Values()
    {
        return Values(data_rowmajor_);
    }

    std::vector<T> Values(bool row_major)
    {
        if (data_rowmajor_ != row_major)
        {
            WARNING_TRACE("The required order is inconsistent with the data layout order!!!");
        }

        const size_t total_size = this->data_.size();
        std::vector<T> values;
        values.reserve(total_size);
        
        if (!row_major/* || this->data_.is_contiguous()*/)
		{
            /*** 直接内存拷贝优化路径 ***/
            values.assign(this->data_.mem, this->data_.mem + total_size);
        }
		else 
		{
            for (size_t r = 0; r < data_.n_slices; r++)
			{
                const arma::Mat<T> channel = data_.slice(r).t();  /*** 显式拷贝避免悬空引用 ***/
                values.insert(values.end(), channel.begin(), channel.end());
            }
        }
        
        return values;
    }

    /**
     * 以常量1初始化张量
     */
    void Ones() { Fill( T{1} ); }
    
    /**
     * 以随机值初始化张量
     */
    void Rand() { data_.randn(); }
    
    /**
     * 打印张量
     */
    void Show()
    {
        INFO_TRACE("rows=" << data_.n_rows << " cols=" << data_.n_cols << " chns=" << data_.n_slices);
	    for (uint32_t r = 0; r < this->channels(); r++)
		{
            // print data_.slice(r);
            INFO_TRACE("Channel: " << r << "\n" << this->data_.slice(r));
        }
    }

    /**
     * 张量的实际尺寸大小的Reshape pytorch兼容
     * @param shapes 张量的实际尺寸大小
     * @param row_major 根据行主序还是列主序进行reshape
     */
    void Reshape(const TensorShape& shapes)
    {
        /*** 需要保证元素总数量一致 ***/
        if (shapes.Size() != rawdata_size_.Size()) return;

        std::vector<T> values = Values(data_rowmajor_);

        /*** 创建目标尺寸张量 ***/
        ArmaCubeType flattened_cube(shapes.Rows(), shapes.Cols(), shapes.Chns());
        this->data_ = std::move(flattened_cube);
        this->rawdata_size_ = TensorShape(shapes.Rows(), shapes.Cols(), shapes.Chns());

        this->Fill(values, data_rowmajor_);
    }

    void Reshape(const TensorShape& shapes, bool row_major)
    {
	    /*** 需要保证元素总数量一致 ***/
    	if(shapes.Size() != rawdata_size_.Size()) return;
    	
        std::vector<T> values = Values(row_major);  // NOTE
        if (data_rowmajor_ != row_major)
        {
            WARNING_TRACE("The required order is inconsistent with the data layout order!!!");
        }
	    
        /*** 创建目标尺寸张量 ***/
        ArmaCubeType flattened_cube(shapes.Rows(), shapes.Cols(), shapes.Chns());
	    this->data_ = std::move(flattened_cube);
	    this->rawdata_size_ = TensorShape(shapes.Rows(), shapes.Cols(), shapes.Chns());
	    
	    this->Fill(values, row_major);
    }

    /**
     * 展开张量
     */
    void Flatten()
    {
        Flatten(data_rowmajor_);
    }

    void Flatten(bool row_major)
    {
        //ArmaCubeType flattened_cube(this->size(), 1, 1);  /*** 获取原始维度信息 ***/
        if (data_rowmajor_ != row_major)
        {
            WARNING_TRACE("The required order is inconsistent with the data layout order!!!");
        }
		
        const uint32_t rows = this->data_.n_rows;
        const uint32_t cols = this->data_.n_cols;
        const uint32_t channels = this->data_.n_slices;
        const uint32_t total_elems = rows * cols * channels;
        
        /*** 创建目标一维张量 ***/
        ArmaCubeType flattened_cube(total_elems, 1, 1);
        T* dest_ptr = flattened_cube.memptr();
        
        if (row_major)
		{
            /*** 行优先模式处理逻辑 ***/
            for (uint32_t r = 0; r < channels; ++r)
		    {
                /*** 转置矩阵获取行优先数据视图 ***/
                //ArmaCubeType transposed_view = this->data_.slice(r).t();
                ArmaMatType transposed_view = this->data_.slice(r).t();
                
                /*** 计算内存偏移量 ***/
                const uint32_t offset = r * rows * cols;
                
                /*** 批量复制转置后的数据 ***/
                std::copy(transposed_view.memptr(),
                         transposed_view.memptr() + transposed_view.n_elem,
                         dest_ptr + offset);
            }
        }
		else
		{
            /*** 列优先模式直接内存拷贝 ***/
            for (uint32_t r = 0; r < channels; ++r)
		    {
                //const ArmaCubeType& channel = this->data_.slice(r);
                const ArmaMatType& channel = this->data_.slice(r);
                const uint32_t offset = r * rows * cols;
                
                /*** 直接复制列优先存储的原始数据 ***/
                std::copy(channel.memptr(),
                         channel.memptr() + channel.n_elem,
                         dest_ptr + offset);
            }
        }
        
        /*** 使用移动语义更新数据 ***/
        this->data_ = std::move(flattened_cube);
		this->rawdata_size_ = TensorShape(total_elems, 1, 1);
    }

    /**
     * 对张量中的元素进行过滤
     * @param filter 过滤函数
     */
    void Transform(const std::function<T(T)>& filter) { data_.transform(filter); }

    /**
     * 返回数据的原始指针
     * @return 返回数据的原始指针
     */
    const T* rawptr() const {return data_.memptr(); }
    T* rawptr_offset(uint32_t offset) {return data_.memptr() + offset; }  /*** 偏移offset个元素 ***/
    T* rawptr_matrix(uint32_t index) { return rawptr() + (index * rows() * cols()); }  /*** 偏移index个channel ***/

public:
    /**
     * 对张量进行相加
     * @param tenor1 张量1
     * @return 
     */
    Tensor<T> operator+(const Tensor<T>& other) const
    {
        // 获取形状并计算广播形状
        const auto shape_a = this->shapes();
        const auto shape_b = other.shapes();
        auto broadcast_shape = BroadCastedShape(shape_a, shape_b);
        ArmaCubeType result_data(broadcast_shape.Rows(), broadcast_shape.Cols(), broadcast_shape.Chns());
        result_data.zeros();

        if (shape_a == shape_b)
        {
            result_data = std::move(data() + other.data());
        }
        else
        {
            // 预计算广播条件
            const bool a_row_needbroad = shape_a.Rows() == 1;
            const bool a_col_needbroad = shape_a.Cols() == 1;
            const bool a_chn_needbroad = shape_a.Chns() == 1;

            const bool b_row_needbroad = shape_b.Rows() == 1;
            const bool b_col_needbroad = shape_b.Cols() == 1;
            const bool b_chn_needbroad = shape_b.Chns() == 1;

            /*** 多线程并行化外层循环（rows和depth）***/
#pragma omp parallel for collapse(2)
            for (size_t row = 0; row < broadcast_shape.Rows(); ++row) {
                for (size_t chn = 0; chn < broadcast_shape.Chns(); ++chn) {
                    const size_t a_row = (!a_row_needbroad) * row;
                    const size_t a_channl = (!a_chn_needbroad) * chn;
                    const size_t b_row = (!b_row_needbroad) * row;
                    const size_t b_channl = (!b_chn_needbroad) * chn;

                    /*** 内层循环（cols）连续访问，支持SIMD ***/
                    for (size_t col = 0; col < broadcast_shape.Cols(); ++col) {
                        const size_t a_col = (!a_col_needbroad) * col;
                        const size_t b_col = (!b_col_needbroad) * col;
                        result_data(row, col, chn) = data_(a_row, a_col, a_channl) + other.data_(b_row, b_col, b_channl);
                    }
                }
            }

        }

        return Tensor<T>(std::move(result_data));
    }

    /**
     * 对张量进行相减
     * @param tenor1 张量1
     * @return 
     */
    Tensor<T> operator-(const Tensor<T>& other) const
    {
        // 获取形状并计算广播形状
        const auto shape_a = this->shapes();
        const auto shape_b = other.shapes();
        auto broadcast_shape = BroadCastedShape(shape_a, shape_b);
        ArmaCubeType result_data(broadcast_shape.Rows(), broadcast_shape.Cols(), broadcast_shape.Chns());
        result_data.zeros();

        if (shape_a == shape_b)
        {
            result_data = std::move(data() - other.data());
        }
        else
        {
            // 预计算广播条件
            const bool a_row_needbroad = shape_a.Rows() == 1;
            const bool a_col_needbroad = shape_a.Cols() == 1;
            const bool a_chn_needbroad = shape_a.Chns() == 1;

            const bool b_row_needbroad = shape_b.Rows() == 1;
            const bool b_col_needbroad = shape_b.Cols() == 1;
            const bool b_chn_needbroad = shape_b.Chns() == 1;

            /*** 多线程并行化外层循环（rows和depth）***/
#pragma omp parallel for collapse(2)
            for (size_t row = 0; row < broadcast_shape.Rows(); ++row) {
                for (size_t chn = 0; chn < broadcast_shape.Chns(); ++chn) {
                    const size_t a_row = (!a_row_needbroad) * row;
                    const size_t a_channl = (!a_chn_needbroad) * chn;
                    const size_t b_row = (!b_row_needbroad) * row;
                    const size_t b_channl = (!b_chn_needbroad) * chn;

                    /*** 内层循环（cols）连续访问，支持SIMD ***/
                    for (size_t col = 0; col < broadcast_shape.Cols(); ++col) {
                        const size_t a_col = (!a_col_needbroad) * col;
                        const size_t b_col = (!b_col_needbroad) * col;
                        result_data(row, col, chn) = data_(a_row, a_col, a_channl) - other.data_(b_row, b_col, b_channl);
                    }
                }
            }
        }

        return Tensor<T>(std::move(result_data));
    }

    /**
     * 对张量进行矩阵乘法
     * @param tenor1 张量1
     * @return 
     */
    Tensor<T> operator*(const Tensor<T>& other) const
    {
        if (channels() != other.channels() || cols() != other.rows())
        {
            throw std::invalid_argument("通道数不一致");
        }

        ArmaCubeType result_data(rows(), other.cols(), channels());
        result_data.zeros();

#pragma omp parallel for //schedule(dynamic, 4)
        for (size_t r = 0; r < channels(); r++)
        {
            //result_data.slice(r) = data_slice(r) * other.data_slice(r);
            //result_data.slice(r) = std::move(data_slice(r) * other.data_slice(r));
            //data_slice(r)* other.data_slice(r);
            auto& tmp = data_slice(r)* other.data_slice(r);
            result_data.slice(r) = tmp;
            //result_data.slice(r) = std::move(tmp);

        }
        return Tensor<T>(std::move(result_data));
    }

    /**
     * 对张量进行矩阵元素乘法(点乘)
     * @param tenor1 张量1
     * @param pad_default_value 自动广播时填充的默认值
     * @return
     */
    Tensor<T> operator%(const Tensor<T>& other/*, T pad_default_value = 1*/) const
    {
        // 获取形状并计算广播形状
        const auto shape_a = this->shapes();
        const auto shape_b = other.shapes();
        auto broadcast_shape = BroadCastedShape(shape_a, shape_b);
        ArmaCubeType result_data(broadcast_shape.Rows(), broadcast_shape.Cols(), broadcast_shape.Chns());
        result_data.fill(1);

        if (shape_a == shape_b)
        {
            result_data = std::move(data() % other.data());
        }
        else
        {
            // 预计算广播条件
            const bool a_row_needbroad = shape_a.Rows() == 1;
            const bool a_col_needbroad = shape_a.Cols() == 1;
            const bool a_chn_needbroad = shape_a.Chns() == 1;

            const bool b_row_needbroad = shape_b.Rows() == 1;
            const bool b_col_needbroad = shape_b.Cols() == 1;
            const bool b_chn_needbroad = shape_b.Chns() == 1;


            /*** 多线程并行化外层循环（rows和depth）***/
#pragma omp parallel for collapse(2)
            for (size_t row = 0; row < broadcast_shape.Rows(); ++row) {
                for (size_t chn = 0; chn < broadcast_shape.Chns(); ++chn) {
                    const size_t a_row = (!a_row_needbroad) * row;
                    const size_t a_channl = (!a_chn_needbroad) * chn;
                    const size_t b_row = (!b_row_needbroad) * row;
                    const size_t b_channl = (!b_chn_needbroad) * chn;

                    /*** 内层循环（cols）连续访问，支持SIMD ***/
                    for (size_t col = 0; col < broadcast_shape.Cols(); ++col) {
                        const size_t a_col = (!a_col_needbroad) * col;
                        const size_t b_col = (!b_col_needbroad) * col;
                        result_data(row, col, chn) = data_(a_row, a_col, a_channl) * other.data_(b_row, b_col, b_channl);
                    }
                }
            }

        }

        return Tensor<T>(std::move(result_data));
    }

    /**
     * 对张量进行矩阵元素除法(点除)
     * @param tenor1 张量1
     * @param pad_default_value 自动广播时填充的默认值
     * @return
     */
    Tensor<T> operator/(const Tensor<T>& other/*, T pad_default_value = 1*/) const
    {
        // 获取形状并计算广播形状
        const auto shape_a = this->shapes();
        const auto shape_b = other.shapes();
        auto broadcast_shape = BroadCastedShape(shape_a, shape_b);
        ArmaCubeType result_data(broadcast_shape.Rows(), broadcast_shape.Cols(), broadcast_shape.Chns());
        result_data.fill(1);

        if (shape_a == shape_b)
        {
            result_data = std::move(data() / other.data());
        }
        else
        {
            // 预计算广播条件
            const bool a_row_needbroad = shape_a.Rows() == 1;
            const bool a_col_needbroad = shape_a.Cols() == 1;
            const bool a_chn_needbroad = shape_a.Chns() == 1;

            const bool b_row_needbroad = shape_b.Rows() == 1;
            const bool b_col_needbroad = shape_b.Cols() == 1;
            const bool b_chn_needbroad = shape_b.Chns() == 1;


            /*** 多线程并行化外层循环（rows和depth）***/
#pragma omp parallel for collapse(2)
            for (size_t row = 0; row < broadcast_shape.Rows(); ++row) {
                for (size_t chn = 0; chn < broadcast_shape.Chns(); ++chn) {
                    const size_t a_row = (!a_row_needbroad) * row;
                    const size_t a_channl = (!a_chn_needbroad) * chn;
                    const size_t b_row = (!b_row_needbroad) * row;
                    const size_t b_channl = (!b_chn_needbroad) * chn;

                    /*** 内层循环（cols）连续访问，支持SIMD ***/
                    for (size_t col = 0; col < broadcast_shape.Cols(); ++col) {
                        const size_t a_col = (!a_col_needbroad) * col;
                        const size_t b_col = (!b_col_needbroad) * col;
                        //result_data(row, col, chn) = (other.data_(b_row, b_col, b_channl)==0) ? T(0) : (data_(a_row, a_col, a_channl) / other.data_(b_row, b_col, b_channl));
                        result_data(row, col, chn) = data_(a_row, a_col, a_channl) / other.data_(b_row, b_col, b_channl);
                    }
                }
            }

        }

        return Tensor<T>(std::move(result_data));
    }

    /**
     * 对张量进行比较
     * @param tenor1 张量1
     * @param tensor2 张量2
     * @note 绝对差异（"absdiff"）、相对差异（"reldiff"）
     * @return 
     */
    bool appro_equal(const Tensor<T>& other, float threshold = 1e-5f) const
    {
        if (shapes() != other.shapes()) return false;
        return arma::approx_equal(data(), other.data(), "absdiff", threshold);
    }

    /**
     * 填充张量
     * @param pads 填充张量的尺寸
     * @param padding_value 填充张量
     * @note: pytorch中填充模式有几种，比如'constant'、'reflect'、'replicate'、'circular'等。
     *        'constant'是填充固定值，
     *        'reflect'是镜像反射边界，
     *        'replicate'是复制边缘值，
     *        'circular'则是循环填充，常用于周期信号
     * 本函数实现为固定值填充，且目前只对row和col两个维度进行padding
     */
    Tensor<T> Paddingd(const std::vector<uint32_t>& pads, T padding_value) const
    {
        /*** 原始维度 ***/
        const uint32_t rows = this->data_.n_rows;
        const uint32_t cols = this->data_.n_cols;
        const uint32_t channels = this->data_.n_slices;
    
        /*** 填充尺寸 ***/
        const uint32_t pad_rows1 = pads.at(0);  /*** 上方填充行数 ***/
        const uint32_t pad_rows2 = pads.at(1);  /*** 下方填充行数 ***/
        const uint32_t pad_cols1 = pads.at(2);  /*** 左侧填充列数 ***/
        const uint32_t pad_cols2 = pads.at(3);  /*** 右侧填充列数 ***/
    
        /*** 填充后的大小 ***/
        const uint32_t new_rows = rows + pad_rows1 + pad_rows2;
        const uint32_t new_cols = cols + pad_cols1 + pad_cols2;
    
        /*** 创建填充后的cube并初始化 ***/
        ArmaCubeType padded_cube(new_rows, new_cols, channels);
        padded_cube.fill(padding_value);  /*** 先全部按照padding_value填充整个Tensor ***/
    
        /*** 目标区域坐标计算 ***/
        const uint32_t row_start = pad_rows1;
        const uint32_t row_end = row_start + rows - 1;
        const uint32_t col_start = pad_cols1;
        const uint32_t col_end = col_start + cols - 1;
        for (uint32_t r = 0; r < channels; ++r) /*** 逐通道填充 ***/
        {
            /*** 原始数据区域 ***/
            padded_cube.slice(r).submat(row_start, col_start, row_end, col_end) = this->data_.slice(r);
        }
    
        return Tensor<T>(std::move(padded_cube));
    }

private:
    TensorShape BroadCastedShape(const TensorShape& shape_a, const TensorShape& shape_b) const
    {
        /*** 确保输入为三维 ***/
        if (shape_a.Dims() != 3 || shape_b.Dims() != 3)
        {
            throw std::invalid_argument("广播仅支持三维张量");
        }

        auto BroadCastSingleDim = [](uint32_t dim_a, uint32_t dim_b)->uint32_t {
            if (dim_a == dim_b)
            {
                return dim_a;
            }
            else if (dim_a == 1)
            {
                return dim_b;
            }
            else if (dim_b == 1)
            {
                return dim_a;
            }
            else
            {
                throw std::invalid_argument("形状不兼容，无法广播");
            }
        };

        TensorShape broadcast_shape;
        broadcast_shape.setChns(BroadCastSingleDim(shape_a.Chns(), shape_b.Chns()) );
        broadcast_shape.setCols(BroadCastSingleDim(shape_a.Cols(), shape_b.Cols()) );
        broadcast_shape.setRows(BroadCastSingleDim(shape_a.Rows(), shape_b.Rows()) );
        return broadcast_shape;
    }


private:
    ArmaCubeType data_;  /*** 张量数据 ***/
	TensorShape rawdata_size_;  /*** 张量数据的实际尺寸大小(张量数据大小可能比Tensor小) ***/
    bool data_rowmajor_ = false;  /*** 数据是否按照行优先分布的 ***/
};

//using ftensor = Tensor<float>;
//using sftensor = std::shared_ptr<Tensor<float>>;


}  // namespace kuiper_infer 


#endif //__ZINFER_TENSOR_H__
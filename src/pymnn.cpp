#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <string>
#include <memory>
#include <sstream>

namespace py = pybind11;

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<double>(buf1.size);

    py::buffer_info buf3 = result.request();

    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *ptr2 = static_cast<double *>(buf2.ptr);
    double *ptr3 = static_cast<double *>(buf3.ptr);

    for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        ptr3[idx] = ptr1[idx] + ptr2[idx];
    return result;
}

class pyMNN
{
public:
    pyMNN(const std::string& model_path, const std::string& input_name, const std::vector<std::string>& output_names){
        auto interpreter_ptr = MNN::Interpreter::createFromFile(model_path.c_str());
        if(interpreter_ptr == nullptr){
            throw std::runtime_error("can not load model " + model_path);
        }

        _interpreter.reset(interpreter_ptr);
        MNN::ScheduleConfig cfg;
        _session = _interpreter->createSession(cfg);
        _input_ts = _interpreter->getSessionInput(_session, input_name.c_str());
        _input_shape = _input_ts->shape();
        // _input_ts->printShape();
        _output_names = output_names; 
    }

    ~pyMNN(){}

    std::unordered_map<std::string, py::array> Infer(py::array_t<float> input_array){
        py::buffer_info bufinfo = input_array.request();
        if(bufinfo.ndim != 4 || _input_shape.size() != 4){
            throw std::runtime_error("input dimentions should be 4");
        }
        for(int dim = 0; dim < 4; ++dim){
            if(_input_shape[dim] != bufinfo.shape[dim])
            {
                std::stringstream msg;
                msg << "input shape should be equal with MNN mdoel. desired shape "<< \
                    _input_shape[0] << "x"<<_input_shape[1]<<"x"<<_input_shape[2]<<"x"<<_input_shape[3];
                throw std::runtime_error(msg.str());
            }
        }
        auto input_ts = MNN::Tensor::create(_input_ts->shape(), halide_type_of<float>(), bufinfo.ptr, MNN::Tensor::CAFFE );
        _input_ts->copyFromHostTensor(input_ts);
        _interpreter->runSession(_session); 

        // return map<string, numpy.array>
        std::unordered_map<std::string, py::array> result_dict;
        for(auto outname : _output_names){
            MNN::Tensor* out_ts = _interpreter->getSessionOutput(_session, outname.c_str());
            MNN::Tensor host_ts(out_ts, MNN::Tensor::CAFFE);
            out_ts->copyToHostTensor(&host_ts);
            std::vector<ssize_t> strides;
            for(int ii = 0; ii < host_ts.dimensions(); ++ii){
                strides.push_back(host_ts.stride(ii) * sizeof(float));
            } 
            result_dict[outname] = py::array(py::buffer_info(
                host_ts.host<float>(),                           /* data as contiguous array  */
                sizeof(float),                          /* size of one scalar        */
                py::format_descriptor<float>::format(), /* data type                 */
                host_ts.dimensions(),                                    /* number of dimensions      */
                host_ts.shape(),                                   /* shape of the matrix       */
                strides                                  /* strides for each axis     */
            ));
        }
        return result_dict;
    }


private:
    std::unique_ptr<MNN::CV::ImageProcess> _preprocess;
    std::unique_ptr<MNN::Interpreter> _interpreter;
    MNN::Session* _session;
    MNN::Tensor* _input_ts;
    std::vector<int> _input_shape;
    std::vector<std::string> _output_names;
};


PYBIND11_MODULE(pymnn, m) {
    m.doc() = "MNN python wrapper";
    // m.def("add_arrays", &add_arrays, "Add two NumPy arrays");

    py::class_<pyMNN>(m, "pyMNN")
    .def(py::init<std::string, std::string, std::vector<std::string> >())
    .def("Infer", &pyMNN::Infer)
    .def("__repr__",
        [](const pyMNN &a) {
            return "<pyMNN repr>";
        });
}
std::string get_program_text(void);
cl_device_id create_device(void);
cl_program build_program(cl_context ctx, cl_device_id dev);

/*
Функция инициализации DEVICE
*/
cl_device_id create_device(void) 
{
    cl_platform_id platform;
    cl_device_id dev;
    cl_int err = 0;
    err |= clGetPlatformIDs(1, &platform, NULL);
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND) 
    {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err) throw;
    return dev;
}

/*
Функция для получения текста программы 
файла *.cl
*/
std::string get_program_text() 
{
  std::ifstream t("/home/galahad/Documents/5_course/2_semester/paralcomput/lab2/gpu_opencl/include/kernels.cl");
  return std::string((std::istreambuf_iterator<char>(t)),
                      std::istreambuf_iterator<char>());
}

/*
Функция для компиляции программы ядер
Аргументы:
    - ctx - контекст объект OpenCL (cl_context)
    - dev - ID устройства на котором будут происходить 
      вычисления (cl_device_id)
*/
cl_program build_program(cl_context ctx, cl_device_id dev) 
{
    int err;

    std::string src = get_program_text();
    std::cout << src.size() <<std::endl;
    const char* src_text = src.data();
    size_t src_length = src.size();

    cl_program program = clCreateProgramWithSource(ctx, 1, &src_text, &src_length, &err);

    printf("lCreateProgramWithSource error = %d\n", err);

    err |= clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    return program;
}

/*
Функция для вычисления глобального размера
кратного размеру рабочей группы
Аргументы:
    - x - глобальный размер
    - y - размер рабочей группы
*/
int align(int x, int y) 
{
    int result = ((x + y - 1) / y) * y;
    return result;
}

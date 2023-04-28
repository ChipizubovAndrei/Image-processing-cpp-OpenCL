// Библиотека для работы с изображениями jpg
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb/stb_image.h"
#include "lib/stb/stb_image_write.h"

// Подключение библиотек
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <fstream>

// OpenCL includes
#include <CL/cl.h>

#include "include/functions.hpp"

int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    char path_to_image1 [] = "../images/airplane3.jpg";
    char path_to_image1_4 [] = "../images/airplane3_4.jpg";
    char path_to_save [] = "../output/output3_4.jpg";

    cl_uchar* src;
    int srcH, srcW, ch;

    cl_uchar* src_4;
    int srcH_4, srcW_4, ch_4;

    unsigned char* dst;

    // Загрузка изображений
    src = stbi_load( path_to_image1, &srcW, &srcH, &ch, 3 );
    src_4 = stbi_load( path_to_image1_4, &srcW_4, &srcH_4, &ch_4, 3 );

    // Расчет размеров выходного изображения
    int dstH = 4*srcH_4;
    int dstW = 4*srcW_4;

    // Выделение памяти для выходного изображения
    dst = (unsigned char*)malloc(dstH*dstW*3 * sizeof( unsigned char ));

    // Проверка на подключение OpenCL
    cl_int CL_err = CL_SUCCESS;

    // Поиск доступных устройств
    cl_uint numPlatforms = 0;

    CL_err = clGetPlatformIDs( 0, NULL, &numPlatforms );

    if (CL_err == CL_SUCCESS)
        printf("%u platform(s) found\n", numPlatforms);
    else
        printf("clGetPlatformIDs(%i)\n", CL_err);

    auto t1 = high_resolution_clock::now();

    // Выбор устройства для вычисления
    cl_int err;
    cl_device_id device = create_device();

    // Создание контекста — объекта, отвечающего за конкретную сессию взаимодействия с устройством.
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // Компиляция OpenCL ядра
    cl_program program = build_program(context, device);

    // Загрузка необходимого ядра
    cl_kernel kernel = clCreateKernel(program, "CrossScaleSubtraction", &err);
    printf("clCreateKernel Error = %d\n", err);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    printf("clCreateCommandQueue Error = %d\n", err);

    // Выделение памяти на GPU
    cl_mem dev_src = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof( cl_uchar )*srcW*srcH*3, NULL, NULL);
    cl_mem dev_src_4 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof( cl_uchar )*srcW_4*srcH_4*3, NULL, NULL);
    cl_mem dev_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof( unsigned char )*dstW*dstH*3, NULL, NULL);

    // Копирование массивав с HOST на DEVICE
    clEnqueueWriteBuffer(queue, dev_src, 0, 0,  sizeof( cl_uchar )*srcW*srcH*3, src, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, dev_src_4, 0, 0,  sizeof(cl_uchar)*srcW_4*srcH_4*3, src_4, 0, NULL, NULL);

    // Передача аргументов ядру
    err = 0;
    err |= clSetKernelArg( kernel, 0, sizeof( cl_mem ), &dev_src );
    err |= clSetKernelArg( kernel, 1, sizeof( cl_mem ), &dev_src_4 );
    err |= clSetKernelArg( kernel, 2, sizeof( cl_mem ), &dev_dst );
    err |= clSetKernelArg( kernel, 3, sizeof( int ), &srcH );
    err |= clSetKernelArg( kernel, 4, sizeof( int ), &srcW );
    err |= clSetKernelArg( kernel, 5, sizeof( int ), &srcH_4 );
    err |= clSetKernelArg( kernel, 6, sizeof( int ), &srcW_4 );

    const int n_dims = 2;
    int ItemPerGroup = 32;
    size_t local_size[n_dims] = { ItemPerGroup , ItemPerGroup };
    size_t global_size[n_dims] = { align(srcH_4, local_size[0]), align(srcW_4, local_size[1] ) };

    // запускаем двумерную задачу
    err |= clEnqueueNDRangeKernel(queue, kernel, n_dims, NULL, 
                                  global_size, local_size, 0, NULL, NULL);

    if (err != 0)
    {
        printf("Programm crashed. Error code: %d\n", err);
    }

    // Копирования массива с DEVICE на HOST
    err |= clEnqueueReadBuffer(queue, dev_dst, CL_TRUE, 0, 
                                sizeof( unsigned char )*dstH*dstW*3, dst, 0, NULL, NULL);
    printf("clEnqueueReadBuffer Error = %d\n", err);

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    printf("Time = %f\n", ms_double);

    // ждём завершения всех операций
    clFinish(queue);

    // Освобождение памяти DEVICE
    clReleaseKernel(kernel);
    clReleaseMemObject(dev_src);
    clReleaseMemObject(dev_src_4);
    clReleaseMemObject(dev_dst);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    // Сохранение изображения
    int err_stb = stbi_write_jpg( path_to_save, dstW, dstH, ch, dst, (dstW*ch ));

    if (err_stb == 0)
    {
        printf("\nWrite/Read error\n");
    }

    // Освобождение памяти HOST 
    stbi_image_free( src );
    stbi_image_free( src_4 );
    free( dst );

    return 0;
}
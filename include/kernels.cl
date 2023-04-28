__kernel void GaussUpsample(__global unsigned char* src, __global unsigned char* dst, int srcH, int srcW);
__kernel void Convolution2D(__global unsigned char* src, __global unsigned char* dst, 
                              int srcH, int srcW);
__kernel void CrossScaleSubtraction(__global unsigned char* src, __global unsigned char* src_4, 
                                    __global unsigned char* dst, int srcH, int srcW, int srcH_4, int srcW_4);

/*
Функция для вычисления кроссмаштабного
вычитания матриц
Аргументы:
    - src - исходное изображение
    - src_4 - изображение уменьшенное в 4 раза
    - dst - возвращаемое изображение
    - srcH - высота исходного изображения
    - srcW - ширина исходного изображения
    - srcH_4 - высота изображения уменьшенного в 4 раза
    - srcW_4 - ширина изображения уменьшенного в 4 раза
*/
__kernel void CrossScaleSubtraction(__global unsigned char* src, __global unsigned char* src_4, 
                                    __global unsigned char* dst, int srcH, int srcW, int srcH_4, int srcW_4)
{
    int scale = 4;
    // Вычисление размеров выходного изображения
    int dstH = scale*srcH_4;
    int dstW = scale*srcW_4;

    // Получение глобального ID Item
    int yid = get_global_id(0);
    int xid = get_global_id(1);

    // Получение размера Work Space
    int ygid = get_global_size(0);
    int xgid = get_global_size(1);

    while (yid < srcH_4)
    {
        while (xid < srcW_4)
        {
            for (int ch = 0; ch < 3; ch++)
            {
                for (int ky = 0; ky < scale; ky++)
                {
                    for (int kx = 0; kx < scale; kx++)
                    {
                        int sy = (int)(yid*scale + ky);
                        int sx = (int)(xid*scale + kx);
                        if (sy >= 0 && sy < dstH && sx >= 0 && sx < dstW)
                        {
                            // Проверка, что разность пикселей > 0
                            if (((int)src[(sy*dstW + sx)*3 + ch] - (int)src_4[(yid*srcW_4 + xid)*3 + ch]) < 0)
                            {
                                dst[(sy*dstW + sx)*3 + ch] = 0;
                            }
                            else
                            {
                                dst[(sy*dstW + sx)*3 + ch] = (unsigned char)((int)src[(sy*dstW + sx)*3 + ch] - (int)src_4[(yid*srcW_4 + xid)*3 + ch]);
                            }
                        }
                    }
                }
            }
            xid += xgid;
        }
        yid += ygid;
    }
}

/*
Функция для вычисления свертки
Аргументы:
    - src - исходное изображение
    - dst - возвращаемое изображение
    - srcH - высота исходного изображения
    - srcW - ширина исходного изображения
*/
__kernel void Convolution2D(__global unsigned char* src, __global unsigned char* dst, 
                              int srcH, int srcW)
{
    // Задание параметров свертки
    int kernelYX = 4;
    int m_pad = 0;
    int dilation = 1;
    int stride = 2;
    // Вычисление размеров выходного изображения
    int dstH = (int)((srcH + 2 * m_pad - dilation * (kernelYX - 1) - 1) / stride + 1);
    int dstW = (int)((srcW + 2 * m_pad - dilation * (kernelYX - 1) - 1) / stride + 1);

    // Получение глобального ID Item
    int yid = get_global_id(0);
    int xid = get_global_id(1);

    // Получение размера Work Space
    int ygid = get_global_size(0);
    int xgid = get_global_size(1);

    int m_value;

    while (yid < dstH)
    {
        while (xid < dstW)
        {
            // Перебор по фильтрам (выходным каналам)
            for (int ch = 0; ch < 3; ch++)
            {
                // Проход по ядру
                int sum = 0;
                for (int ky = 0; ky < kernelYX; ky++)
                {
                    for (int kx = 0; kx < kernelYX; kx++)
                    {
                        int sy = (int)(yid*stride - m_pad + ky);
                        int sx = (int)(xid*stride - m_pad + kx);
                        if (sy >= 0 && sy < srcH && sx >= 0 && sx < srcW)
                        {
                            sum += src[(sy*srcW + sx)*3 + ch];
                        }
                    }
                }
                m_value = sum / (kernelYX*kernelYX);
                dst[(yid*dstW + xid)*3 + ch] = (unsigned char)m_value;
            }
            xid += xgid;
        }
        yid += ygid;
    }
}

/*
Функция для создания изображения с добавленными строками
Аргументы:
    - src - исходное изображение
    - dst - возвращаемое изображение
    - srcH - высота исходного изображения
    - srcW - ширина исходного изображения
*/
__kernel void GaussUpsample(__global unsigned char* src, __global unsigned char* dst, int srcH, int srcW)
{
    // Получение глобального ID Item
    int ygid = get_global_size(0);
    int xgid = get_global_size(1);

    // Получение размера Work Space
    int yid = get_global_id(0);
    int xid = get_global_id(1);

    // Вычисление выходного размера изображения
    int dstH = srcH + 2;
    int dstW = srcW + 2;

    while( yid < srcH )
    {
        while (xid < srcW)
        {
            // Угловые пиксели
            // Левый верхний угол
            if (yid == 0 && xid == 0)
            {
                dst[(yid*dstW + xid)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[(yid*dstW + xid)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[(yid*dstW + xid)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 1)*dstW + xid)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
                
                dst[(yid*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[(yid*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[(yid*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 1)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
                
            }
            // Правый верхний угол
            else if (yid == 0 && xid == srcW - 1)
            {
                dst[(yid*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[(yid*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[(yid*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 1)*dstW + xid + 2)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 2)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 2)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
                
                dst[((yid + 1)*dstW + xid + 2)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 2)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 2)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 1)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
            }
            // Левый нижний угол
            else if (yid == srcH - 1 && xid == 0)
            {
                dst[((yid + 1)*dstW + xid)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 2)*dstW + xid)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 2)*dstW + xid)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 2)*dstW + xid)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
                
                dst[((yid + 2)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 2)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 2)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 1)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
            }
            // Правый нижний угол
            else if (yid == srcH - 1 && xid == srcW - 1)
            {
                dst[((yid + 1)*dstW + xid + 2)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 2)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 2)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 2)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 2)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 2)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
                
                dst[((yid + 2)*dstW + xid + 2)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 2)*dstW + xid + 2)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 2)*dstW + xid + 2)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 1)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
            }
            // Граничные пиксели
            // Левая граница
            else if (yid != 0 && xid == 0)
            {
                dst[((yid + 1)*dstW + xid)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 1)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
            }
            // Верхняя граница
            else if (yid == 0 && xid != 0)
            {
                dst[((yid)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 1)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
            }
            // Нижняя граница
            else if (yid == srcH - 1 && xid != 0)
            {
                dst[((yid + 2)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 2)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 2)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 1)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
            }
            // Правая граница
            else if (yid != 0 && xid == srcW - 1)
            {
                dst[((yid + 1)*dstW + xid + 2)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 2)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 2)*3 + 2] = src[(yid*srcW + xid)*3 + 2];

                dst[((yid + 1)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
            }
            else
            {
                dst[((yid + 1)*dstW + xid + 1)*3 + 0] = src[(yid*srcW + xid)*3 + 0];
                dst[((yid + 1)*dstW + xid + 1)*3 + 1] = src[(yid*srcW + xid)*3 + 1];
                dst[((yid + 1)*dstW + xid + 1)*3 + 2] = src[(yid*srcW + xid)*3 + 2];
            }
            xid += xgid;
        }
        yid += ygid;
    }
}
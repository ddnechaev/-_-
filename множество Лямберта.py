import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

# Параметры изображения
width, height = 800, 800
max_iter = 100

# Инициализация OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Код функции ядра
kernel_code = """
#include <pyopencl-complex.h>

__kernel void lambert_set(
    __global int *output, 
    const int width, 
    const int height, 
    const int max_iter)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float real = 4.0f * (x - width / 2) / (0.5f * width);
    float imag = 4.0f * (y - height / 2) / (0.5f * height);
    cfloat_t z = cfloat_new(real, imag);
    cfloat_t one = cfloat_new(1.0f, 0.0f);

    int iteration = 0;
    for (int i = 0; i < max_iter; i++) {
        if (cfloat_abs(z) > 1000) break;  // Escape if diverges
        z = cfloat_sub(z, cfloat_exp(cfloat_neg(z)));
        iteration++;
    }

    output[y * width + x] = iteration;
}
"""

# Компилируем программу
program = cl.Program(context, kernel_code).build()

# Создание выходного буфера
output = np.zeros(width * height, dtype=np.int32)
output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output.nbytes)

# Запуск ядра
program.lambert_set(queue, (width, height), None, output_buffer, np.int32(width), np.int32(height), np.int32(max_iter))
cl.enqueue_copy(queue, output, output_buffer)

output = output.reshape((height, width))
plt.imshow(output, cmap='hot', extent=(-2, 2, -2, 2))
plt.colorbar()
plt.title("Lambert Set")
plt.show()

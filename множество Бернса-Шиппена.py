import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

# Параметры изображения
width, height = 800, 800
max_iter = 256

# Инициализация OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Код функции ядра
kernel_code = """
__kernel void burns_ship(
    __global int *output, 
    const int width, 
    const int height, 
    const int max_iter,
    const float zoom,
    const float offsetX,
    const float offsetY)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float jx = 1.5f * (x - width / 2) / (0.5f * zoom * width) + offsetX;
    float jy = (y - height / 2) / (0.5f * zoom * height) + offsetY;

    float zx = jx;
    float zy = jy;
    int iteration = 0;

    while (zx * zx + zy * zy < (2 * 2) && iteration < max_iter) {
        float xtemp = zx * zx - zy * zy + jx;
        zy = fabs(2 * zx * zy) + jy;
        zx = fabs(xtemp);
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
program.burns_ship(queue, (width, height), None, output_buffer, np.int32(width), np.int32(height), np.int32(max_iter),
                   np.float32(0.5), np.float32(-0.5), np.float32(-0.5))
cl.enqueue_copy(queue, output, output_buffer)

output = output.reshape((height, width))
plt.imshow(output, cmap='hot', extent=(-2, 1, -1.5, 1.5))
plt.colorbar()
plt.title("Фрактал множества Бернса-Шиппена")
plt.show()

import pyopencl as cl
import numpy as np

# Параметры изображения
width, height = 800, 800
max_iter = 50

# Инициализация OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Код функции ядра
kernel_code = """
__kernel void newton_fractal(
    __global int *output, 
    const int width, 
    const int height, 
    const int max_iter)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float jx = 3.0f * (x - width / 2) / (0.5f * width);
    float jy = 3.0f * (y - height / 2) / (0.5f * height);

    float zx = jx;
    float zy = jy;
    int iteration = 0;
    float tx, ty, xtemp, norm;

    while (iteration < max_iter) {
        norm = zx*zx + zy*zy;
        tx = (2*norm + zx) * zx / (3*norm);
        ty = (norm - zx) * zy / (3*norm);

        xtemp = tx;
        zx -= (zx * zx * zx - 3 * zx * zy * zy - 1) / (3 * (zx * zx + zy * zy));
        zy -= (3 * zx * zx * zy - zy * zy * zy) / (3 * (zx * zx + zy * zy));

        if ((zx - tx)*(zx - tx) + (zy - ty)*(zy - ty) < 1e-5) break;
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
program.newton_fractal(queue, (width, height), None, output_buffer, np.int32(width), np.int32(height),
                       np.int32(max_iter))
cl.enqueue_copy(queue, output, output_buffer)

# Вывод изображения с помощью matplotlib
import matplotlib.pyplot as plt

output = output.reshape((height, width))
plt.imshow(output, cmap='hot')
plt.colorbar()
plt.title("Фрактал множества Ньютона f(z) = z^3 - 1")
plt.show()

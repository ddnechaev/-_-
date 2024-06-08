import pyopencl as cl
import numpy as np

# Инициализация OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Код функции ядра
kernel_code = """
inline float lcg_random(uint *seed) {
    const uint a = 1664525;
    const uint c = 1013904223;
    const uint m = 0xffffffff;
    *seed = (a * (*seed) + c) & m;
    return (float)(*seed) / (float)m;
}

__kernel void monte_carlo_pi(__global int *count, const unsigned int n) {
    int gid = get_global_id(0);
    uint seed = 123456789 + gid * 17;  // Простой способ различать seed для разных gid
    float x, y;
    int local_count = 0;

    for (int i = 0; i < n; i++) {
        x = lcg_random(&seed);
        y = lcg_random(&seed);
        if ((x * x + y * y) <= 1.0f) {
            local_count++;
        }
    }
    atomic_add(count, local_count);
}
"""

# Компилируем OpenCL программу
program = cl.Program(context, kernel_code).build()

# Количество испытаний и размер буфера
num_points_per_work_item = 10000
num_work_items = 256
buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, num_work_items * np.dtype(np.int32).itemsize)

# Запуск функции ядра
program.monte_carlo_pi(queue, (num_work_items,), None, buffer, np.uint32(num_points_per_work_item))

# Чтение результата
result = np.zeros(num_work_items, dtype=np.int32)
cl.enqueue_copy(queue, result, buffer)
total_in_circle = np.sum(result)

# Вычисление Pi
pi_estimate = 4.0 * total_in_circle / (num_points_per_work_item * num_work_items)
print(f"Estimated Pi = {pi_estimate}")

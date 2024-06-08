import pyopencl as cl
import numpy as np

# Дифференциальное уравнение: dy/dt = -2y + sin(t)
# Начальное условие: y(0) = 1

# Создаем контекст и очередь
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Программа на OpenCL
kernel_code = """
__kernel void runge_kutta_step(
    __global const float *y_old, 
    __global float *y_new, 
    const float dt, 
    const float t)
{
    int gid = get_global_id(0);
    float k1 = -2.0f * y_old[gid] + sin(t);
    float k2 = -2.0f * (y_old[gid] + 0.5f * dt * k1) + sin(t + 0.5f * dt);
    float k3 = -2.0f * (y_old[gid] + 0.5f * dt * k2) + sin(t + 0.5f * dt);
    float k4 = -2.0f * (y_old[gid] + dt * k3) + sin(t + dt);
    y_new[gid] = y_old[gid] + dt * (k1 + 2.0f * (k2 + k3) + k4) / 6.0f;
}
"""

# Компилируем программу
program = cl.Program(context, kernel_code).build()

# Начальные условия
y0 = np.array([1.0], dtype=np.float32)
y = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=y0)
dt = np.float32(0.01)  # шаг по времени
t_final = 10.0
t = 0.0

# Расчет
while t < t_final:
    program.runge_kutta_step(queue, y0.shape, None, y, y, dt, np.float32(t))
    t += dt
    cl.enqueue_copy(queue, y0, y)

print(f"y({t_final}) = {y0[0]}")

import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Создаем контекст и очередь выполнения
    platform = cl.get_platforms()[0]  # Выбираем первую платформу
    device = platform.get_devices()[0]  # Выбираем первое устройство
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Код на языке OpenCL C для фрактала множества Кантора
    kernel_code = """
    __kernel void generate_cantor(__global int* output, const int depth) {
        int idx = get_global_id(0);
        int step = 1 << depth; // Размер шага на каждом уровне рекурсии
        int pos = idx * step;
        for (int i = 0; i < depth; i++) {
            int section = pos / (3 << i);
            if (section % 3 == 1) {
                output[idx] = 0; // Пропускаем среднюю треть, помечаем как отсутствующую
                return;
            }
        }
        output[idx] = 1; // Маркируем присутствие части фрактала
    }
    """

    # Компиляция программы
    program = cl.Program(context, kernel_code).build()

    # Создаем буфер на устройстве
    output = np.zeros(1000, dtype=np.int32)  # Должен быть достаточно большим, чтобы учесть все элементы
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output.nbytes)

    # Запуск ядра
    depth = 5  # Глубина рекурсии множества Кантора
    global_size = (1000,)  # Размер должен соответствовать размеру массива output
    program.generate_cantor(queue, global_size, None, output_buffer, np.int32(depth))

    # Чтение результата
    cl.enqueue_copy(queue, output, output_buffer)
    queue.finish()

    # Выводим результат
    print("Результат генерации множества Кантора:")
    print(output)

    # Визуализация результата
    visualize_cantor(output, depth)

def visualize_cantor(data, depth):
    plt.figure(figsize=(10, 6))
    y = 0
    step = len(data)
    for i in range(depth):
        x_coords = np.where(data[:step] == 1)[0] * (1 / step)
        y_coords = [y] * len(x_coords)
        plt.scatter(x_coords, y_coords, color='black', s=1)  # используем маленькие точки для представления
        y += 1
        step //= 3
    plt.title('Визуализация множества Кантора')
    plt.yticks(range(depth), labels=[f'Итерация {i+1}' for i in range(depth)])
    plt.xlabel('Позиция')
    plt.ylabel('Итерации')
    plt.show()

if __name__ == "__main__":
    main()

import time

import numpy as np
import matplotlib.pyplot as plt
import numerical_methods_Volt_II_Fred_II as nm
import pandas as pd
import math
import sympy as sym
from sympy import *


def Test_quadrature_method_Volt_II():
    # функция для представления результатов в виде таблицы с использованием библиотеки pandas
    # где x1,x2 - массивы узлов равномерной сетки y1,y2 - массивы значений в этих узлах
    def printTable(x1, y1, x2, y2):
        n = int((len(x2) - 1) / (len(x1) - 1))
        s1 = pd.Series(x1)
        s2 = pd.Series(y1)
        s3 = pd.Series([x2[i] for i in range(0, len(x2), n)])
        s4 = pd.Series([y2[i] for i in range(0, len(y2), n)])
        return pd.DataFrame(
            {"x1": s1, f"y при n={len(y1)}"
            : s2, "x2": s3, f"y при n={len(y2)}": s4}).to_string()

    # Реализуем функцию представление резутата в виде таблице для одного набора значений
    def printTableOne(x1, y1):
        n = int(len(x1) / 5)
        s1 = pd.Series([x1[i] for i in range(0, len(x1), n)])
        s2 = pd.Series([y1[i] for i in range(0, len(y1), n)])
        return pd.DataFrame(
            {"x1": s1, f"y при n={len(y1)}"
            : s2}).to_string()
    def Test1_Volt_II_Rect_vect_linspace():
        #задача решить уравнение численным методом квадратур
         # y(x)-\int_{0}^{x}e^{-(x-s)}y(s)ds=e^{-x} x\in[0,1]
         # c точным решением y(x)=1

        # функция ядра уравнения(1)
        def K(x, s):
            return np.exp(-(x - s))

        # функция правой части уравнения (1)
        def F(x):
            return np.exp(-x)

        def Exact_solution(a, b, n):
            return np.ones(n)
        # вызов функции Volt_II_Rect с параметрами
        a = 0
        b = 1
        n1 = 5
        n2 = 21
        print("Volt_II_Rect")
        y1, x1 = nm.Volt_II_Rect_vect_linspace(K, F, a, b, n1)
        y2, x2 = nm.Volt_II_Rect_vect_linspace(K, F, a, b, n2)
        # представление результата
        print(printTable(x1, y1, x2, y2))
        # построение графиков
        plt.plot(x1, y1, linestyle=' ', marker='o', label='n=5')
        plt.plot(x2, y2, linestyle=' ', marker='s', label='n=21')
        plt.plot(x1, [1] * len(x1), label='y=1')
        plt.xlabel("x")
        plt.ylabel('y')
        plt.legend(["n=5", "n=21", "точное решение"])
        plt.grid(True)
        plt.show()
        nm.HDyPlot(nm.Volt_II_Rect_vect_linspace, Exact_solution, K, F, 0, 1, 5, 11)
        # С помощью этого сценария проведем два численных эксперимента:
        # найдем приближенные решения уравнения на сетках с шагом h = 0.05
        # и h = 0.25.

    def Test2_Volt_II_Rect_vect_linspace():
        #задача решить уравнение численным методом квадратур
        #y(x)=e^{x^{2}}+\int_{0}^{x}e^{x^2-s^2}y(s)ds, x\in[0,1].
        # # c точным решением y(x) = e^{x^{2}+x}
        # функция правой части f(x) для уравнения (6)
        def F(x):
            return np.exp(x * x)

        # функция ядра уравнения(6)
        def K(x, s):
            return np.exp(x * x - s * s)

        # фунция точного решения для уравнения (6)
        def Exact_solution(a, b, n):
            x = np.linspace(a, b, n)
            return np.array(np.exp(x * x + x))

            # Вызов функции Volt_II_Rect_vect_linspace

        y1, x1 = nm.Volt_II_Rect_vect_linspace(K, F, 0, 1, 5)
        y2, x2 = nm.Volt_II_Rect_vect_linspace(K, F, 0, 1, 21)
        # представление результата приближенного решения
        print("Volt_II_Rect_vect_linspace")
        print(printTable(x1, y1, x2, y2))
        # представление точного решения
        print("\nТочное решение")
        print(printTable(x1, Exact_solution(0, 1, 5), x2, Exact_solution(0, 1, 21)))
        # построение графиков
        plt.plot(x1, y1, linestyle=' ', marker='o', label='n=5')
        plt.plot(x2, y2, linestyle=' ', marker='s', label='n=21')
        plt.plot(x2, np.exp(x2 * x2 + x2), label='y=e^(x*x+x)')
        plt.xlabel("x")
        plt.ylabel('y')
        plt.legend(["n=5", "n=21", "Точное решение"])
        plt.grid(True)
        plt.show()
        nm.HDyPlot(nm.Volt_II_Rect_vect_linspace,Exact_solution,K,F,0,1)

    def Test_Volt_II_Rect_vect():
        # задача решить уравнение численным методом квадратур на неравномерной сетке
        # y(x)=(1-xe^{2x})cos 1 -e^{2x}sin1+\int_0^x(1-(x-s)e^{2x})y(s)ds x \in [0, 2.5]
        # # c точным решением y(x) = e^{x}(cos(e^{x})−e^{x}sin(e^{x}))

        # реализуем не равномерное разбиение сетки с концетрацией узлов к левому краю
        def To_left(a, b, n):
            def MesH(x):
                return np.exp(x) - 1

            pseudo_b = math.log(b + 1)
            pseudo_a = math.log(a + 1)
            pseudo_x = np.linspace(pseudo_a, pseudo_b, n)
            return MesH(pseudo_x)

        # реализуем не равномерное разбиение сетки с концетрацией узлов к правому краю
        def To_right(a, b, n):
            def MesH(x):
                return (-np.exp(-x) + a + 1) * (b - a + 1)

            pseudo_b = -math.log(a + 1 - b / (b - a + 1))
            pseudo_a = -math.log(a + 1 - a / (b - a + 1))
            pseudo_x = np.linspace(pseudo_a, pseudo_b, n)
            return MesH(pseudo_x)
        # функция правой части f(x)
        def F(x):
            return (1 - x * np.exp(2 * x)) * np.cos(1) - np.exp(2 * x) * np.sin(1)
        # функция ядра уравнения
        def K(x, s):
            return 1 - (x - s) * np.exp(2 * x)
        # функция точного решения уранения (7)
        def Exact_solution(a, b, n, rasp_n):  # np.linspace
            x = rasp_n(a, b, n)
            return np.array(np.exp(x) * (np.cos(np.exp(x)) - np.exp(x) * np.sin(np.exp(x))))
            # разбиение которое сконцентрированно к левому концу

        YL, XL = nm.Volt_II_Rect_vect(K, F, 0, 2.5, 101, To_left)
        # разбиение которое сконцентрированно к правому концу
        YR, XR = nm.Volt_II_Rect_vect(K, F, 0, 2.5, 101, To_right)
        # равномерное разбиение
        YLn, XLn = nm.Volt_II_Rect_vect(K, F, 0, 2.5, 101, np.linspace)

        print("точное решение с распределением к левому краю")
        print(printTableOne(XL, Exact_solution(0, 2.5, 101, To_left)))
        print("точное решение с распределением к правому краю")
        print(printTableOne(XR, Exact_solution(0, 2.5, 101, To_right)))
        print("точное решение с равномерным распределением")
        print(printTableOne(np.linspace(0, 2.5, 101), Exact_solution(0, 2.5, 101, np.linspace)))
        print("решение с распределением к правому краю")
        print(printTableOne(XL, YL))
        print("решение с распределением к левому краю")
        print(printTableOne(XR, YR))
        print("решение с равномерным распределением ")
        print(printTableOne(XLn, YLn))
        # постироим график для сравнения точного решения и приближенного
        # с распределением к правому краю
        plt.plot(XR, YR, linestyle=' ', marker='s', label='n=21')
        plt.plot(To_right(0, 2.5, 101), Exact_solution(0, 2.5, 101, To_right), label='y=res')
        plt.xlabel("x")
        plt.ylabel('y')
        plt.legend(["n=101", "точное решение"])
        plt.grid(True)
        plt.show()

        # построим график зависимости
        # нормы относильной ошибки решение приближенного метода
        # от количества узлов сетки для разных распределений
        print("график ошибки для разных распределений")
        dy1, n1 = nm.NDyPlot(nm.Volt_II_Rect_vect, Exact_solution, K, F, 0, 2.5, 51, 102, To_right, False)
        dy2, n2 = nm.NDyPlot(nm.Volt_II_Rect_vect, Exact_solution, K, F, 0, 2.5, 51, 102, To_left, False)
        dy3, n3 = nm.NDyPlot(nm.Volt_II_Rect_vect, Exact_solution, K, F, 0, 2.5, 51, 102, np.linspace, False)
        plt.plot(n1, dy1)
        plt.plot(n2, dy2)
        plt.plot(n3, dy3)
        plt.xlabel("n")
        plt.ylabel('||dy||')
        plt.legend(["распределения к правому краю",
                    "распределения к левому краю", "равномерного распределения"])
        plt.grid(True)
        plt.show()

    def Test_Volt_II_Rect_Simpson_linspace():
        # задача решить уравнение численным методом квадратур
        # y(x)=e^{x^{2}}+\int_{0}^{x}e^{x^2-s^2}y(s)ds, x\in[0,1].
        # # c точным решением y(x) = e^{x^{2}+x}

        # функция точного решения для уравнения (6)
        def Exact_solution(a, b, n, rasp_n):
            x = rasp_n(a, b, n)
            return np.exp(x * x + x)
        # функция правой части f(x) для уравнения (6)
        def F(x):
            return np.exp(x * x)
        # функция ядра  уравнения (6) K(x,s)
        def K(x, s):
            return np.exp(x * x - s * s)
        # обёртка для метода NdyPlot
        # добавляется параметр rasp_n который не на что не влияет
        def Volt_II_Rect_Simpson_stub(K, f, a, b, n, rasp_n):
            return nm.Volt_II_Rect_Simpson_linspace(K, f, a, b, n)
            # Вычисляем приближеннымы методами трапеций и симсона
        Y11, X11 = nm.Volt_II_Rect_vect(K, F, 0, 1, 5, np.linspace)
        Y22, X22 = nm.Volt_II_Rect_vect(K, F, 0, 1, 21, np.linspace)
        Y1, X1 = nm.Volt_II_Rect_Simpson_linspace(K, F, 0, 1, 5)
        Y2, X2 = nm.Volt_II_Rect_Simpson_linspace(K, F, 0, 1, 21)
        # изуализируем результат выполнения функции Volt_II_Rect_vect
        print("Volt_II_Rect_vect")
        print(printTable(X11, Y11, X22, Y22))
        # изуализируем результат выполнения функции Volt_II_Rect_Simpson_linspace
        print("Volt_II_Rect_Simpson_linspace")
        print(printTable(X1, Y1, X2, Y2))
        # изуализируем результат выполнения функции Exact_solution
        print(" exp(X2 * X2 + X2)")
        print(printTable(X1, np.exp(X1 * X1 + X1), X2, np.exp(X2 * X2 + X2)))

        # Построим график решений при n=21 для визуального представления точности приближенного
        # и точного решения
        plt.plot(X2, Y2, linestyle=' ', marker='s', label='n=21')
        plt.plot(X22, Y22, linestyle=' ', marker='D', label='n=21')
        plt.plot(X2, np.exp(X2 * X2 + X2), label='y=e^(x*x+x)')
        plt.xlabel("x")
        plt.ylabel('y')
        plt.legend(["Метод Симпсона", "Метод трапеций", "точное решение"])
        plt.grid(True)
        plt.show()
        # Построим график решений при n=5 для визуального представления точности приближенного
        # и точного решения
        # plt.plot(X1, Y1, linestyle=' ', marker='s', label='n=5')
        plt.plot(X11, Y11, linestyle=' ', marker='D', label='n=5')
        plt.plot(X1, np.exp(X1 * X1 + X1), label='y=e^(x*x+x)')
        plt.xlabel("x")
        plt.ylabel('y')
        plt.legend(["Метод Симпсона", "Метод трапеций", "точное решение"])
        plt.grid(True)
        plt.show()
        # Nplot можно использовать для функции Volt_II_Rect_Simpson только на равномерной сетке
        # для этого используем обёртку Volt_II_Rect_Simpson_stub
        er_sim, n_er_sim = nm.NDyPlot(Volt_II_Rect_Simpson_stub, Exact_solution, K, F, 0, 1,
                                 start_n=5, end_n=41, rasp_n=np.linspace, printgraf=False)

        er_trap, n_er_trap = nm.NDyPlot(nm.Volt_II_Rect_vect, Exact_solution, K, F, 0, 1,
                                        start_n=5, end_n=41, rasp_n=np.linspace, printgraf=False)
        # строим график нормы относильной ошибки решение приближенного метода
        # от количества узлов сетки n
        plt.plot(n_er_sim, er_sim)
        plt.plot(n_er_trap, er_trap)
        plt.xlabel("n")
        plt.ylabel('||dy||')
        plt.legend(["Метод Симпсона", "Метод трапеций"])
        plt.grid(True)
        plt.show()

    def Test_Nonlin_Volt_II():
        # задача решить уравнение численным методом квадратур
        #y(x)=1-x+\int_0^x\left[x e^{s(x-2 s)}+e^{-2 s^2}\right] y^2(s) d s
        # # c точным решением y(x) = e^{x*x}

        a = 0
        b = 0.1
        n = 12
        def K(x, s):
            return x * np.exp(s * (x - 2 * s)) + np.exp(-2 * s ** 2)
        def f(x):
            return 1 - x

        y, x = nm.Nonlin_Volt_II_Rect_linspace(a, b, n, K, f)
        u_exact = np.exp(x ** 2)

        plt.plot(x, y, 'or', label='Приближенное решение')
        plt.plot(x, u_exact, 'b', label='Точное решение')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Приближенное и точное решение')
        plt.legend()
        plt.grid(True)
        plt.show()

def Test_simpleIter():
    # дополним функцианалом метод HDyPlot для использования итерационных методов
    # довавим к существующей реализации возможность построить
    # график зависимости количества итераций k от длинны шага h
    def HDyPlotIter(func, exact_solution, Kfunc, f, a, b, tol, start_n=5, end_n=41):
        h = list()
        morm_error = list()
        k = list()
        for n in range(start_n, end_n):
            # приближенное решение
            y, x, iter = func(a, b, n, Kfunc, f, tol)
            # добавление в масив h длинну шага равномерной сетки
            h.append((b - a) / (n - 1))
            # вектор отклонения от точного решения
            residual = np.array(exact_solution(a, b, n) - y)
            # Добавление в массив Y норму вектора ошибки
            morm_error.append(math.sqrt((np.dot(residual, residual))
                                        / (np.dot(exact_solution(a, b, n), exact_solution(a, b, n)))))
            # добавление в масив k количество итераций метода
            k.append(iter)
        # построим график ошибки от длинны шага
        plt.plot(h, morm_error, label='h')
        plt.xlabel("h")
        plt.ylabel('||dy||')
        plt.legend(["Ошибка"])
        plt.grid(True)
        plt.show()
        # построим график количества итераций от длинны шага
        plt.plot(h, k, label='k')
        plt.xlabel("h")
        plt.ylabel('k')
        plt.legend(["Количество итераций"])
        plt.grid(True)
        plt.show()



    def Test1_Volt_II_SimpleIter():
        # задача решить уравнение численным методом простой итерации
        # y(x)=1+\int_{0}^{x}y(s)ds,x \in[0,7]
        # # c точным решением y(x) = e^{x}


        # функция правой части f(x) для уравнения
        def f(x):
            return np.ones(len(x))

        # функция ядра  уравнения K(x,s)
        def K(x, s):
            return 1

        a = 0
        b = 7
        n = 101
        tol = 1e-03

        # точное решение уравнения
        def Exact_solution(x):
            return np.exp(x)

        # c помощью функции Volt_II_SimpleIter вычисяем приближенное решение уравнения (4)
        y, x, iter = nm.Volt_II_SimpleIter_linspace(K, f, a, b, n, tol)
        # строим график y(x) точного и приближенного решения
        plt.plot(x, y, linestyle=' ', marker='o', label='n=101')
        plt.plot(x, Exact_solution(x), label='res')
        plt.xlabel("x")
        plt.ylabel('y')
        plt.legend(["n=101", "точное решеник"])
        plt.grid(True)
        plt.show()


        HDyPlotIter(nm.Volt_II_SimpleIter_linspace, Exact_solution, K, f, a, b, tol)

    def Test2_Volt_II_SimpleIter():
        # задача решить уравнение численным методом квадратур
        # y(x)=1+\int_{0}^{x}y(s)ds,x \in[0,7]
        # # c точным решением y(x) = e^{x}

        # функция правой части f(x) для уравнения
        def f(x):
            return x

        # функция ядра  уравнения  K(x,s)
        def K(x, s):
            return (s - x)

        a = 0
        b = 2 * math.pi
        n = 101
        tol = 1e-03

        # точное решение уравнения
        def Exact_solution(x):
            return np.sin(x)

        # находим решение уравнения (5) методом простой итерации
        y, x, iter = nm.Volt_II_SimpleIter_linspace(a, b, n, K, f, tol)
        plt.plot(x, y, linestyle=' ', marker='o', label='n=101')
        plt.plot(x, Exact_solution(x), label='res')
        plt.xlabel("x")
        plt.ylabel('y')
        plt.legend(["n=101", "точное решеник"])
        plt.grid(True)
        plt.show()

    def Test_Volt_II_SimpleIter_With_Simpson():
        # задача решить уравнение численным методом квадратур
        # y(x)=1+\int_{0}^{x}y(s)ds,x \in[0,7]
        # # c точным решением y(x) = e^{x}

        # функция правой части уравнения (4)
        def f(x):
            return np.ones(len(x))

        # функция ядра уравнения(4)
        def K(x, s):
            return 1

        a = 0
        b = 7
        n = 101
        tol = 1e-03

        # фунция точного решения для уравнения (4)
        def Exact_solution(x):
            return np.exp(x)

        # находим решение уравнения (4) на равномерной сетке [0,7] методом простой итерации
        y, x, iter = nm.Volt_II_SimpleIter_With_Simpson_linspace(a, b, n, K, f, tol)
        # строим график приближенного и точного решения
        plt.plot(x, y, linestyle=' ', marker='o', label='n=101')
        plt.plot(x, Exact_solution(x), label='res')
        plt.xlabel("x")
        plt.ylabel('y')
        plt.legend(["n=101", "точное решеник"])
        plt.grid(True)
        plt.show()
        print(iter)
        print("")

        # переопределения метода необходимо для использования HPlot
        def Exact_solution(a, b, n):
            x = np.linspace(a, b, n)
            return np.exp(x)

        HDyPlotIter(nm.Volt_II_SimpleIter_With_Simpson_linspace, Exact_solution, K, f, a, b, tol)

    def Test_Nonlin_VoltII_SimpleIter():
        # задача решить уравнение численным методом квадратур
        # y(x) = \int_0^x\frac{1+ y^2(s)}{1 + s^2}ds, x ∈ [0, 10].
        # # c точным решением y(x) = x

        # запишем подынтегральную функцию как единую функцию F(x,s,y(x))
        def F(x, s, y):
            return (1 + y ** 2) / (1 + s ** 2)

        # функция правой части f(x) для уравнения
        def f(x):
            return x * 0;

        # фунция точного решения для уравнения
        def Exact_solution(x):
            return x

        a = 0
        b = 10
        n = 101
        tol = 1e-03

        # вычисляем приближенное решение методом простой итерации
        y, x, iter = nm.Nonlin_Volt_II_SimpleIter_linspace(F, f, a, b, n, tol)
        # строим графики приближенног8о и точного решения
        plt.plot(x, y, linestyle=' ', marker='o', label='n=101')
        plt.plot(x, Exact_solution(x), label='res')
        plt.xlabel("x")
        plt.ylabel('y')
        plt.legend(["n=101", "точное решеник"])
        plt.grid(True)
        plt.show()

def Test_Fred_II():

    def Test1_fred_II_trapez_linspace():
        # задача решить уравнение численным методом квадратур
        # y(x) =25−16sin^2(x)+ \frac{3}/{10\pi}\int_{-pi}^{pi}\frac{y(s)}{0.64\cos^2 (\frac{x+s}{2})-1}ds, x ∈ [0, 10].
        # # c точным решением y(x) = 17/2 + (128/17)\cos(2x)
        # Определение ядра уравнения K(x, s)
        def K(x, s):
            # Функция возвращает значение ядра для заданных x и s
            return 1 / (0.64 * np.cos((x + s) / 2) * np.cos((x + s) / 2) - 1)

        # Определение правой части уравнения f(x)
        def f(x):
            # Функция возвращает значение правой части уравнения для заданного x
            return 25 - 16 * np.sin(x) * np.sin(x)

        # Определение точного решения
        def Exact_solution(x):
            # Функция возвращает точное решение уравнения для заданного x
            return 17 / 2 + (128 / 17) * np.cos(2 * x)

        # Задание границы интервала интегрирования и параметра lambda
        a = -np.pi
        b = np.pi
        Lambda = 3 / (10 * np.pi)
        n = 37
        y, x = nm.fred_II_trapez_linspace(K, f, Lambda, a, b, n)
        # Построение графика решения и точного решения
        plt.plot(x, y, 'ro', label='Приближенное решение')
        plt.plot(x, Exact_solution(x), 'g', label='Точное решение')
        plt.xlabel('x')  # Подпись оси x
        plt.ylabel('y')  # Подпись оси y
        plt.grid(True)
        plt.legend()
        plt.show()

    def Test2_fred_II_trapez_linspace():
        # задача решить уравнение численным методом квадратур
        # y(x) =25−16sin^2(x)+ \frac{3}/{10\pi}\int_{-pi}^{pi}\frac{y(s)}{0.64\cos^2 (\frac{x+s}{2})-1}ds, x ∈ [0, 10].
        # # c точным решением y(x) = 17/2 + (128/17)\cos(2x)
        # Определение ядра уравнения K(x, s)

        # Определяем функции K(x, s) и f(x) соответственно
        def K(x, s):
            # Функция возвращает значение ядра для заданных x и s
            return x * s

        def f(x):
            # Функция возвращает значение правой части уравнения для заданного x
            return (5 / 6) * x

        # Определяем интервал [a, b] и количество узлов сетки n
        a = 0
        b = 1
        n = 100  # Примерное количество узлов

        # Задаем параметр lambda для метода
        Lambda = 1 / 2

        # Решаем уравнение с помощью метода
        y, x = nm.fred_II_trapez_linspace(K, f, Lambda , a, b, n)

        # Exact_solution(x)=x

        # Строим график приближенного и точного решений
        plt.plot(x, y, label='Приближенное решение', color='blue')
        plt.plot(x, x, label='Точное решение', linestyle='--', color='red')
        plt.xlabel('x')  # Подпись оси x
        plt.ylabel('y(x)')  # Подпись оси y
        plt.title('Сравнение приближенного и точного решений')  # Заголовок графика
        plt.legend()
        plt.grid(True)  # Включение сетки на графике
        plt.show()  # Отображение графика


    def Test_Degenerate_Fred_II():
        # задача решить уравнение численным методом квадратур
        #y(x)-\int_0^1(1+2xs)y(s)ds=-\frac{1}{6}x-\frac{1}{2},\hspace{10mm}x\in[0,1]
        # # c точным решением y(x)=x+\frac{1}{2}.


        x, y = sym.symbols('x  y')
        # Из уравнения мы видим что
        a = 0
        b = 1
        Lambda = 1
        f = -x / 6 - 1 / 2
        # зададим массивы alpha и beta  по формуле K(x,s)=sum(a(s)b(x)) из элементов вырожденого ядра
        # необходимое преобразование s->x для вычисления интегралла
        alpha = sym.Array([1, 2 * x])
        beta = sym.Array([1, x])

        y = nm.Degenerate_Fred_II(alpha, beta, f, a, b, Lambda)
        print("уравнение у(х) имеет вид:")
        sym.init_printing()  # функция для читабельного вывода результатов
        print(y)

    def Test_Degenerate_Fred_II_teylor():
        # задача решить уравнение численным методом вырожденных ядер
        # y(x)=x^2+λ\int_{-1}^1(x+s)y(s)ds,x\in[-1,1]
        #λ^2\neq \frac{3}{4}
        # c точным решением y=x^2+\frac{2λx+4λ^2/3}{4-4λ^2}.

        sym.init_printing()  # функция для читабельного вывода результатов
        x, y, s = symbols('x y s')
        y = nm.Degenerate_Fred_II_teylor(x + s, 2, x * x, -1, 1, 1)
        print('уравнение 1')
        print(y)
        # задача решить уравнение численным методом вырожденных ядер
        # y(x)-\int_0^{1/2}exp(-x^2s^2)y(s)ds=1, x\in[0,1/2].
        # приближенное решение уравнения (15) имеет вид:
        # \widetilde{y}(x)=1.9930-0.0833x^2+0.0007x^4.
        print('уравнение 2')
        y = nm.Degenerate_Fred_II_teylor(exp(-x * x * s * s), 5, 1, 0, 0.5, 1, True)
        print(y)


def Test_Galer_Petrov():
    def Test_GalerkinPetrov_Fred_II():
        # задача решить уравнение численным методом Галеркина - Петрова
        # y(x)=1+\int_{-1}^1\left(x s+x^2\right) y(s) d s, \quad x \in[-1,1] .
        # c точным решением y(x)=1+6 x^2

        # Определим функцию fi_i(x)
        def fi(x, i):
            return x ** i

        # Определим функцию psi_i(x)
        def psi(x, i):
            return x ** (i - 1)

        # Определение ядра K(x, s)
        def K(x, s):
            return x * s + x ** 2

        # Определение функции f(x)
        def f(x):
            return 1

        # Задание границ интервала интегрирования и других параметров
        a = -1
        b = 1
        Lambda = 1
        n = 2

        x, y = nm.GalerkinPetrov_Fred_II(fi, psi, K, f, a, b, Lambda, n)

        # Визуализация результатов
        plt.plot(x, y, label='Приближенное решение')

        # Построение графика точного решения (y(x) = 1 + 6 * x**2)

        Exact_solution = 1 + 6 * x ** 2
        plt.plot(x, Exact_solution, label='Точное решение', linestyle='None', marker='o', color='red')

        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(1, 7)
        plt.xlim(-1, 1)
        plt.grid(True)
        plt.legend()
        plt.show()

def Fred_II_Colloc():
    def Test_Fred_II_Colloc():
        # задача решить уравнение численным методом Галеркина - Петрова
        # y(x)=1+\frac{4}{3} x+\int_{-1}^1\left(x s^2-x\right) y(s) d s, \quad x \in[-1,1] .
        # c точным решением y(x)=1
        # Определение ядра уравнения
        def K(x, s):
            return x * s + x ** 2

        # Определение правой части уравнения
        def f(x):
            return 1 + 0 * x

        # Задание параметров
        n = 3
        m = 20
        lambda_ = 1
        a = -1
        b = 1

        # Засекаем время перед выполнением кода
        start_time = time.time()

        # Вызов функции для решения уравнения Фредгольма
        y, condNumb = Fred_II_Colloc(K, f, a, b, lambda_, n, m)
        # Генерация значений x для построения графика
        x = np.linspace(a, b, m)
        # Определение точного решения
        y_exact = 6 * x ** 2 + 1

        # Построение графика
        plt.plot(x, y, label='Приближенное решение')
        plt.plot(x, y_exact, 'ro', label='Точное решение')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Решение уравнения Фредгольма II')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Определение относительной ошибки решения
        error = np.linalg.norm(y - y_exact, np.inf) / np.linalg.norm(y_exact, np.inf)
        print(f"Относительная ошибка решения: {error}")

        # Засекаем время после выполнения кода
        end_time = time.time()
        # Вычисляем время выполнения
        execution_time = end_time - start_time
        print(f"Время выполнения программы: {execution_time} секунд")

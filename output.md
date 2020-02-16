```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
%matplotlib inline
```

# Нормальное распределение

Вот так можно сгенерировать выборку из нормально распределённой случайной
величины с параметрами $\mu=2.0$ и $\sigma=0.5$:

```python
mu = 2.0
sigma = 0.5

# зададим нормально распределенную случайную величину
norm_rv = sts.norm(loc=mu, scale=sigma)

# сгенерируем 10 значений
norm_rv.rvs(size=10)
```

Параметр ```loc``` задаёт $\mu$, ```scale``` — среднеквадратичное отклонение
$\sigma$, ```size``` — размер выборки. Имя параметра ```size``` при вызове
функции ```rvs``` можно не писать.

Следующая функция возвращает значение функции распределения нормальной случайной
величины в точке, соответствующей её аргументу:

```python
norm_rv.cdf(3)
```

Построим график функции распределения:

```python
x = np.linspace(0,4,100)
cdf = norm_rv.cdf(x) # функция может принимать и вектор (x)
plt.plot(x, cdf)
plt.ylabel('$F(x)$')
plt.xlabel('$x$')
```

А так можно вычислить значение функции плотности вероятности нормального
распределения в заданной точке:

```python
norm_rv.pdf(3)
```

Построим график функции плотности вероятности:

```python
x = np.linspace(0,4,100)
pdf = norm_rv.pdf(x)
plt.plot(x, pdf)

plt.ylabel('$f(x)$')
plt.xlabel('$x$')
```

# Равномерное распределение на отрезке

Вот так можно сгенерировать выборку из случайной величины, имеющей равномерное
распределение на отрезке $[a,b]$:

```python
a = 1
b = 4

# обратите внимание, что в этой функции задается левая граница и масштаб, а не левая и правая границы:
uniform_rv = sts.uniform(a, b-a)

uniform_rv.rvs(10)
```

А так — вычислять значения функций распределения и плотностей:

```python
x = np.linspace(0,5,100)
cdf = uniform_rv.cdf(x)
plt.plot(x, cdf)

plt.ylabel('$F(x)$')
plt.xlabel('$x$')
```

```python
x = np.linspace(0,5,1000)
pdf = uniform_rv.pdf(x)
plt.plot(x, pdf)

plt.ylabel('$f(x)$')
plt.xlabel('$x$')
```

# Распределение Бернулли

Генерация выборок из распределения Бернулли с заданным параметром $p$:

```python
bernoulli_rv = sts.bernoulli(0.7)

bernoulli_rv.rvs(10)
```

# Биномиальное распределение

Генерация выборок из биномиального распределения:

```python
binomial_rv = sts.binom(20, 0.7)
binomial_rv.rvs(10)
```

Первый аргумент функции binom — значение параметра $n$, второй — параметра $p$.

Функция распределения:

```python
x = np.linspace(0,20,21)
cdf = binomial_rv.cdf(x)
plt.step(x, cdf)

plt.ylabel('$F(x)$')
plt.xlabel('$x$')
```

Функция вероятности ```pmf``` для дискретных случайных величин заменяет функцию
плотности ```pdf```:

```python
x = np.linspace(0,20,21)
pmf = binomial_rv.pmf(x)
plt.plot(x, pmf, 'o')

plt.ylabel('$P(X=x)$')
plt.xlabel('$x$')
```

Посмотрим, как ведут себя биномиально распределенные величины при разных
значениях параметров:

```python
x = np.linspace(0,45,46)
for N in [20, 30]:
    for p in [0.2, 0.7]:
        rv = sts.binom(N, p)
        cdf = rv.cdf(x)
        plt.step(x, cdf, label="$N=%s, p=%s$" % (N,p))
plt.legend()
plt.title("CDF (binomial)")

plt.ylabel('$F(X)$')
plt.xlabel('$x$')
```

```python
x = np.linspace(0,45,46)
symbols = iter(['o', 's', '^', '+'])
for N in [20, 30]:
    for p in [0.2, 0.8]:
        rv = sts.binom(N, p)
        pmf = rv.pmf(x)
        plt.plot(x, pmf, next(symbols), label="$N=%s, p=%s$" % (N,p))
plt.legend()
plt.title("PMF (binomial)")

plt.ylabel('$P(X=x)$')
plt.xlabel('$x$')
```

# Распределение Пуассона

Генерация выборок из распределения Пуассона с параметром $\lambda$:

```python
poisson_rv = sts.poisson(5)
poisson_rv.rvs(10)
```

```python
x = np.linspace(0,30,31)
for l in [1, 5, 10, 15]:
    rv = sts.poisson(l)
    cdf = rv.cdf(x)
    plt.step(x, cdf, label="$\lambda=%s$" % l)
plt.legend()
plt.title("CDF (poisson)")

plt.ylabel('$F(x)$')
plt.xlabel('$x$')
```

```python
x = np.linspace(0,30,31)

symbols = iter(['o', 's', '^', '+'])
for l in [1, 5, 10, 15]:
    rv = sts.poisson(l)
    pmf = rv.pmf(x)
    plt.plot(x, pmf, next(symbols), label="$\lambda=%s$" % l)
plt.legend()
plt.title("PMF (poisson)")

plt.ylabel('$P(X=x)$')
plt.xlabel('$x$')
```

# Дискретное распределение общего вида

Чтобы сгенерировать дискретную случайную величину общего вида, нужно задать
множество её значений и соответствующих вероятностей и использовать функцию
```numpy.random.choice```:

```python
elements = np.array([1, 5, 12])
probabilities = [0.05, 0.7, 0.25]
np.random.choice(elements, 10, p=probabilities)
```

# Другие распределения

Существует большое количество других стандартных семейств распределений, многие
из которых также можно генерировать в Питоне.
Например, распределение хи-квадрат $\chi^2_k$, имеющее натуральный параметр $k$,
который называется числом степеней свободы:

```python
x = np.linspace(0,30,100)
for k in [1, 2, 3, 4, 6, 9]:
    rv = sts.chi2(k)
    cdf = rv.cdf(x)
    plt.plot(x, cdf, label="$k=%s$" % k)
plt.legend()
plt.title("CDF ($\chi^2_k$)")
```

```python
x = np.linspace(0,30,100)
for k in [1, 2, 3, 4, 6, 9]:
    rv = sts.chi2(k)
    pdf = rv.pdf(x)
    plt.plot(x, pdf, label="$k=%s$" % k)
plt.legend()
plt.title("PDF ($\chi^2_k$)")
```

Полный список функций SciPy для работы со всеми распределениями можно найти тут:
http://docs.scipy.org/doc/scipy-0.14.0/reference/stats.html

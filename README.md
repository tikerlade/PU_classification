# Positive-Unlabeled learning vs One-Class classification

Пример использовани PU классификации из [статьи](https://www.eecs.wsu.edu/~holder/courses/CptS570/fall09/present/ElkanKDD08.pdf) в качестве тестового задание для стажировки в "JetBrains Research" на лето 2020 года.

- `pu_classifier.py` - содержит реализацию класса PU_classifier

- `PU_classifier.ipynb` - сожержит пример использования классификатора и его тестирование


## Обзор

### Оценки

[](classification_result.jpg)

При обучении получаем следующие метрики (accuracy, усредненный по 100 значениям):

- **PU классификатор:** 0.93

- **LogisticRegression:** 0.96

### Пример
Пример использования классификатора на синтетических данных доступен в [Google Colab](https://drive.google.com/file/d/1wpBSjNfRn3cZCCV_j2ecUEP8DCuYqFNY/view?usp=sharing)


### Данные

Была создана выборка из 1000 элементов, в которой было два класса. Выборка состоит из смеси двух гауссиан. Данные были сгенерированы в двумерном пространстве, для визуализации.

Распределение классов 1/4, т.е. 200 экземпляров положительного класса и 800 экземпляров отрицательного класса.


### Обучение

Для обучения PU классификатора данные были разделены (случайным образом) на маркированные (60 экз. положительного класса) и немаркированные (оставшиеся 940 экз. положительного и отрицательного класса).

Для обучения LogReg классификатора из исходной выборки были выбраны 700 экземляров.


## Итоговый проект Megafon

**Стек**:
**ML**: sklearn, pandas, numpy, lightgbm, boruta

**Данные**:
В качестве исходных данных представлена информация об отклике абонентов на предложение подключения одной из услуг. Каждому пользователю может быть сделано несколько предложений в разное время, каждое из которых он может или принять, или отклонить.

Отдельным набором данных будет являться нормализованный анонимизированный набор признаков, характеризующий профиль потребления абонента. Эти данные привязаны к определенному времени, поскольку профиль абонента может меняться с течением времени.

Данные train и test разбиты по периодам – на train доступно 4 месяцев, а на test отложен последующий месяц.
Итого, в качестве входных данных будут представлены:
data_train.csv: id, vas_id, buy_time, target
features.csv.zip: id, <feature_list>

И тестовый набор:
data_test.csv: id, vas_id, buy_time
- target - целевая переменная, где 1 означает подключение услуги, 0 - абонент не подключил услугу соответственно.
- buy_time - время покупки, представлено в формате timestamp
- id - идентификатор абонента
- vas_id - подключаемая услуга

**Задача**:
Необходимо построить  алгоритм, который для каждой пары пользователь-услуга определит вероятность подключения услуги

**Метрика**:
Невзвешенная f1 (f1_score(..., average='macro') из библиотеки sklearn)

**Репозиторий** содержит:

1. Работающую модель в формате pickle в папке **model**, которая принимает файл data_test.csv из корневой папки и записывает в эту же папку файл answers_test.csv. В этом файле должны находится 4 столбца: buy_time, id, vas_id и target.

2. Jupyter-ноутбук с процессом решения

3. Презентация в формате .pdf
4. Скрипты с функциями обрабоки входных данных и обучения модели в папке **src**


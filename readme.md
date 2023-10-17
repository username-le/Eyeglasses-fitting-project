# Подбор очков

__Идея:__ клиент загружает фото, сервис подбирает знаменитостей в очках, похожих на него (с указанием % сходства)

__Почему?__ По приблизительным оценкам ВОЗ 1.9 миллиардов человек в мире нуждаются в очковой коррекции зрения

__Как?__ 
* Набрана база изображений людей в очках с помощью поисковых запросов ‘cute eyeglasses’, ‘stylish eyeglasses’ и др.

* Библиотека с открытым исходным кодом face_recognition (Автор Adam Geitgey в сотрудничестве с Davis King)

* Натренирована на миллионах изображений знаменитостей из открытых источников

* Функция face_encodings преобразует тестовое изображение в вектор из 128 значений типа float (face vector)
* Составлен список face_encodings для каждого изображения знаменитостей и моделей в модных очках
* При загрузке фото клиента рассчитывается евклидово расстояние для каждого лица из сохраненного списка
* Предъявляем пользователю фотографии похожих людей в оправах, которые возможно подойдут клиенту - в порядке убывания сходства

* Функция compare_faces, которой необходимо передать параметр tolerance (в диапазоне от 0 до 1)
* Чем меньше порог, тем больше сходство

! [$Пример] (https://github.com/username-le/Eyeglasses-fitting-project/blob/main/Chollet.png)

__Итог:__
* Кастомизированное предложение оправы из каталога компании
* Возможность приехать и померять сразу 5-10 отложенных оправ 
* Сразу узнать стоимость и наличие оправы




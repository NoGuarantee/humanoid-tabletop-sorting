# CLAUDE.md

Решение команды 049 для трека манипуляции nssr-polygon: узлы ROS 2 Humble, которые управляют роботом TienKung 2.0 Pro
в симуляторе Isaac Sim. Пул объектов 4 (figure, dish, box). Подход и архитектура: `docs/ANALYSIS.md`,
`docs/ARCHITECTURE.md`, план: `docs/ROADMAP.md`, процесс: `CONTRIBUTING.md`.

## Симулятор лежит в соседнем репозитории

`../nssr-uwt-polygon` (если клон в другом месте, путь указан в `CLAUDE.local.md`). **Только чтение**: решение не
должно зависеть от правок симулятора, на проверке будет оригинал. Связь с ним только через ROS 2.

Где что смотреть (читать по мере надобности, а не целиком):

| Что | Путь в `../nssr-uwt-polygon` |
|---|---|
| Задачи, варианты, правила решения | `hackathon/docs/TASKS.md` |
| Баллы, лимиты времени, формат сдачи | `hackathon/docs/SCORING.md` |
| Топики, типы, номера моторов, кисти, камеры | `hackathon/docs/INTERFACES.md` |
| Пулы объектов по категориям | `hackathon/config/pools.yaml` |
| URDF робота (звенья, суставы, камеры, TCP) | `assets/robots/tiangong_pro/tiangong_pro_with_hands.urdf` |
| Пример узла | `hackathon/ros2/hello_world/hello_world/hello.py` |
| Как проверка ставит и запускает решение | `hackathon/tools/solution.sh` |
| Образ, в котором работает решение | `hackathon/docker/Dockerfile` |

Код симулятора (`source/`, `hackathon/hack_sim/`, `hackathon/ros2/nssr_polygon/`) читать можно, чтобы понять
поведение. Использовать его в решении нельзя.

## Жёсткие правила хакатона

- Положение объектов определяем только по датчикам робота: камеры, глубина, `/joint_states`, `/arm_6dof_*`,
  `/inspire_hand/state/*`, `/tf`. Нельзя читать USD, состояние физики, API Isaac и импортировать код симулятора.
- В `src/` и `run.sh` нельзя обращаться к `/sim/task/check`, `/sim/objects/set`, `/sim/objects/randomize`:
  на проверке они отключены, а обращения к ним фиксируются. Отладочные скрипты кладём в `tools/`.
- Один `run.sh` для всех задач. `/sim/task/done` публикуется в любом исходе.
- В контейнере решения нет файлов симулятора. Всё, что нужно в рантайме (URDF, веса), должно лежать
  в этом репозитории или скачиваться в `setup.sh`.
- Проверяется последний коммит в `main`: туда попадаем только через PR, и `main` всегда должен работать.
- Перед коммитом: `bash tools/check_rules.sh`.

## Соглашения

- Система координат API — `pelvis`, единицы СИ, время только из `/clock` (`use_sim_time: true`).
- Интерфейсы в `src/hack_interfaces` меняем только по договорённости команды: от них зависят все модули.
- Скорость моторов ограничена 0.8 рад/с; приводы недолетают до цели, поэтому контур замыкаем по `/joint_states`.
- Миссия brain идёт в отдельном потоке, узлы крутятся в `MultiThreadedExecutor`. Синхронно ждать внутри callback нельзя.

## Сборка и запуск

ROS 2 на хосте нет, всё работает в Docker из образа `nssr-sim:latest`.

- Симулятор (запускает человек): `cd ../nssr-uwt-polygon && bash hackathon/tools/task.sh sort3 --pool 4`
- ROS-команды к работающему симулятору: `bash ../nssr-uwt-polygon/hackathon/tools/ros.sh ros2 topic list`
- Разработка: `bash tools/dev_shell.sh`, внутри
  `colcon build --symlink-install --build-base /tmp/b --install-base /tmp/i && source /tmp/i/setup.bash`
- Как на проверке: `bash ../nssr-uwt-polygon/hackathon/tools/solution.sh "$PWD"`
- Логику без ROS (task.py, grasp_planner.py) проверяем обычным python3 с заглушками geometry_msgs.

Если docker недоступен (`permission denied`) или симулятор не запущен, не делай выводов о работе кода.
Попроси человека запустить команду и прислать вывод.

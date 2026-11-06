# nssr-sol-team-049

Решение команды 049 для трека манипуляции хакатона nssr-polygon (робот TienKung 2.0 Pro, ROS 2 Humble).
Пул объектов: **4** (figure, dish, box).

> Этот README обязателен по регламенту: его сверяют с поведением решения на итоговой проверке. Держим его в
> актуальном состоянии: раздел «Как устроено решение» обновляем вместе с кодом.

## Статус

| Задача | Статус |
|---|---|
| `sort3`, пул 4 | не реализовано, решение публикует `done` (0 баллов) |
| `sort2`, пул 4 | не реализовано |
| `sort2`, пул 6 (доп.) | не реализовано |
| `tubes easy` / `standard` / `color` / `lying` | не реализовано, пропускается сигналом `done` |

План и распределение работ: [docs/ROADMAP.md](docs/ROADMAP.md).

## Как устроено решение

Решение — конвейер, а не сквозная обучаемая политика. Камеры и глубина дают объекты в системе `pelvis`, планировщик
выбирает объект и захват, обратная кинематика строит движение руки. Почему так: [docs/ANALYSIS.md](docs/ANALYSIS.md).
Узлы, интерфейсы и state machine: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

| Узел | Роль |
|---|---|
| `hack_perception` | сервис `perception/perceive`: RGB-D → объекты, ящики, штативы в `pelvis` |
| `hack_motion` | actions `motion/move_tcp`, `motion/move_joints`, `motion/hand`: IK, замкнутый контур по `/joint_states`, пресеты кистей |
| `hack_brain` | читает `/sim/task`, ведёт цикл «осмотр → выбор → захват → укладка», следит за лимитом времени, публикует `/sim/task/done` |

Решение получает данные только от датчиков робота: камер, глубины, `/joint_states`, `/arm_6dof_*`, `/inspire_hand/state/*`
и `/tf`. Стол, ящики, штативы и шкаф находятся так же, камерами: заранее измеренные координаты сцены в код не зашиваем.
Из топиков задачи используются только `/sim/task` и `/sim/objects/names`.

### Сортировка (`sort2`, `sort3`)

_Заполняется по мере реализации: детекция, оценка положения по глубине, захват, раскладка._

### Пробирки (`tubes`)

_Заполняется по мере реализации._

### Модели и библиотеки

_Список с версиями: см. `requirements.txt`._

## Запуск

Так решение ставится и запускается на итоговой проверке:

```bash
cd /solution
source /opt/ros/humble/setup.bash && colcon build
source install/setup.bash && bash run.sh
```

Проверить у себя, при запущенном симуляторе, из репозитория симулятора:

```bash
bash hackathon/tools/task.sh sort3 --pool 4
bash hackathon/tools/solution.sh ~/путь/к/nssr-sol-team-049      # во втором терминале
```

Цикл разработки, ветки и правила: [CONTRIBUTING.md](CONTRIBUTING.md).

## Структура

```
src/
  hack_interfaces/   msg / srv / action, общие для узлов
  hack_motion/       IK, движения суставов, кисти
  hack_perception/   камеры, глубина, детекция
  hack_brain/        state machine, skills, планировщик захватов, config/grasp_db.yaml
  hack_bringup/      launch-файл
tools/
  check_rules.sh     нет ли в src/ запрещённых обращений (отладочные топики, внутренности симулятора)
  dev_shell.sh       контейнер разработки с этим репозиторием в /solution
docs/
  ANALYSIS.md        разбор задач, баллов, правил и выбор подхода
  ARCHITECTURE.md    узлы, интерфейсы, state machine, отладка модулей
  ROADMAP.md         вехи и роли
run.sh               запуск решения
requirements.txt     pip-зависимости с версиями
```

## Скринкасты

Прикладываются к Releases этого репозитория: по одному на каждую задачу и вариант, окно симулятора от старта до
`done` и терминал со строкой `layout seed`.

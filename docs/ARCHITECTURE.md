# Архитектура решения

Три процесса ROS 2 и общий пакет интерфейсов. Такое разбиение позволяет писать модули параллельно и отлаживать каждый
без остальных. Почему выбран конвейер, а не обучаемая политика, объяснено в [ANALYSIS.md](ANALYSIS.md), раздел 6.

```
             /sim/task, /sim/objects/names
                        │
                        ▼
┌───────────────────────────────────────────┐
│ hack_brain  (оркестратор)                 │
│  • state machine задачи                   │
│  • skills: pick / place                   │
│  • grasp_planner + grasp_db.yaml          │
│  • бюджет времени, /sim/task/done         │
└──────┬─────────────────────────┬──────────┘
       │ srv perception/perceive │ actions motion/move_tcp, move_joints, hand
       ▼                         ▼
┌──────────────────┐    ┌─────────────────────────┐
│ hack_perception  │    │ hack_motion             │
│ камеры+глубина+tf│    │ IK | cuRobo, контур по  │
│ → SceneObjects   │    │ /joint_states, кисти    │
└──────────────────┘    └─────────────────────────┘
       ▲                         │
  /cam_*/..., /tf          /arm|head|waist/cmd_pos, /inspire_hand/ctrl/*
```

| Пакет | Что внутри | Владелец |
|---|---|---|
| `hack_interfaces` | msg, srv, action (ament_cmake) | все; меняется только по договорённости |
| `hack_motion` | `motion_node`, `robot.py` (моторы, пределы, пресеты кистей), `ik.py`; планируется бэкенд cuRobo | motion |
| `hack_perception` | `perception_node`, `geometry.py` (глубина, проекция, tf) | perception |
| `hack_brain` | `brain_node` (state machine), `skills.py`, `grasp_planner.py`, `task.py`, `config/grasp_db.yaml` | brain/infra и grasping |
| `hack_bringup` | `launch/solution.launch.py` | brain/infra |

Почему процесса три. Perception тяжёлый, возможно на GPU, и вызывается **по запросу**: это экономит CPU, а нагрузка
учитывается при равных баллах. Motion держит свой цикл управления и не тормозит из-за нейросетей. Brain — чистая логика.
С бэкендом cuRobo motion тоже работает на GPU, и видеокарту делят perception, motion и сам симулятор.

## Соглашения

- **Система координат API — `pelvis`.** Ноги закреплены, так что `pelvis` неподвижен. Motion сам учитывает текущий угол корпуса.
- Единицы СИ, углы в радианах. Время берём только из `/clock`: у всех узлов `use_sim_time: true`, это задано в launch-файле.
- Perception берёт tf **на метку времени кадра**.
- Отладочные топики (`/sim/task/check`, `/sim/objects/set`, `/sim/objects/randomize`) запрещены в `src/`. Отладочные
  скрипты лежат в `tools/`.

## Интерфейсы

Определения лежат в [src/hack_interfaces](../src/hack_interfaces). Кратко:

| Интерфейс | Тип | Назначение |
|---|---|---|
| `DetectedObject` | msg | `track_id`, `name`, `category`, `cap_color`, `confidence`, `center`, `yaw`, `size`, `top_z`, `lying`, `partially_visible` |
| `Container` | msg | ящик или штатив: `id`, `kind`, `color`, `pose`, `size`, `slots[]`, `slot_occupied[]` |
| `SceneObjects` | msg | `header`, `table_z`, `objects[]`, `containers[]` |
| `perception/perceive` | srv `Perceive` | `camera` (head / torso), `region` (table / hood) → `scene`, `success`, `message` |
| `motion/move_tcp` | action `MoveTcp` | `arm`, `target` (Pose в pelvis), `mode` (direct / linear) → `success`, ошибки, `message` |
| `motion/move_joints` | action `MoveJoints` | `joints[]`, `positions[]`, `speed` → `success`, `max_error` |
| `motion/hand` | action `Hand` | `hand`, `preset`, `force` → `object_detected`, `position[]`, `effort[]` |

Внутри brain используются Python-структуры `GraspCandidate` и `PlacePlan` из
[grasp_planner.py](../src/hack_brain/hack_brain/grasp_planner.py).

Motion обещает остальным модулям:
1. Цели обрезаются по пределам суставов. Если цель недостижима, ответ `unreachable` приходит до начала движения.
2. Суставы приходят к цели одновременно, скорость не выше 0.8 рад/с.
3. Недолёт ПД-регулятора компенсируется по `/joint_states`.

## Планировщик движения

`hack_motion` строит траектории одним из двух бэкендов, бэкенд выбирается параметром `planner`. Зачем cuRobo,
какие у него риски и по каким критериям принимаем пробную интеграцию, описано в [ANALYSIS.md](ANALYSIS.md), раздел 7.3.

| Бэкенд | Что делает | Когда |
|---|---|---|
| `ik` | своя IK (`ik.py`), линейные примитивы, синхронизация суставов | бейзлайн и запасной вариант |
| `curobo` | IK с проверкой коллизий, траектории руки и корпуса вместе | после пробной интеграции |

- Контракт `MoveTcp` и `MoveJoints` от бэкенда не зависит, brain про бэкенд не знает.
- Мир для cuRobo собирается из `SceneObjects`: стол, ящики и соседние объекты как кубоиды, шкаф и штативы по
  геометрии из perception.
- Исполнение: траектория отправляется потоком `cmd_pos` по времени `/clock`, в конце доводка по `/joint_states`.
  Приводы недолетают до цели, поэтому запас по коллизиям 2–3 см.
- Прогрев cuRobo идёт фоном с INIT: лимит отсчитывается с публикации задачи.
- Чтобы планировать сразу на набор кандидатов захвата, в `MoveTcp` понадобится поле `targets[]` и номер выбранной
  цели в ответе. Это изменение интерфейса, только по договорённости команды. Пока его нет, brain перебирает кандидатов
  сам.

## State machine (`brain_node.py`)

Целевая схема. Что из неё уже есть в скелете, см. раздел «Состояние модулей в скелете».

```
INIT ── ждём /sim/task и /clock, считаем дедлайн, ждём perception и motion;
  │       фоном загрузка YOLO и прогрев cuRobo
LOOK ── голова на 0.4 рад вниз, корпус 0 (MoveJoints)
  │
PERCEIVE ── perceive(head, table) → объект, p, (x, y, yaw), OBB; ящики; стол
  │
SELECT ── фильтр: категория из задачи, ящик найден, объект виден целиком, ещё не провален
  │         первыми — поднятые, но лежащие вне ящика (+20 к итогу)
  │         не трогать, если p(объект верен) · P(донесём после подъёма) ≤ 0.5
  │         порядок: ближе к своему ящику; рука: y > 0 → left, иначе right
  │         нет кандидатов → DONE
PLAN ── кандидаты захвата → pelvis → IK и коллизии → лучший кандидат
  │
PICK ── pre_grasp → linear вниз → Hand(preset) → VERIFY: object_detected
  │      → lift → VERIFY: вес в /arm_6dof_*
  │      неудача до подъёма: раскрыть, отойти, следующий кандидат (штрафа нет)
PLACE ── перенос над столом → approach → Hand(open) → retreat → рука пуста?
  │
RESCAN ── объект в ящике? соседей не сдвинули?
  │
BUDGET ── осталось меньше cycle_estimate? → DONE, иначе → LOOK
  │
DONE ── рука пуста, голова и корпус домой, /sim/task/done (в любом исходе, даже при исключении)
```

Пробирки идут по той же схеме, только правило выбора штатива другое (`TaskSpec.hood_racks_for`) и добавляется поворот корпуса
на +1.57 рад. Сейчас это заглушка (`run_tubes`), она сразу ведёт в `DONE`.

**Ловушка rclpy.** Ожидание сервиса или action внутри callback даёт deadlock. Поэтому миссия идёт в отдельном потоке, узел крутится
в `MultiThreadedExecutor`, а [clients.py](../src/hack_brain/hack_brain/clients.py) даёт блокирующие обёртки. Так
и продолжаем.

## Состояние модулей в скелете

| Модуль | Работает | Не реализовано |
|---|---|---|
| motion | `MoveJoints` (синхронизация, компенсация недолёта), `Hand` (пресеты, `object_detected`) | `ik.solve`, режим `linear`, `force`, ошибка TCP по FK, бэкенд cuRobo |
| perception | подписки, снимок кадров, облако точек в `pelvis` (`cloud_in_pelvis`) | `detect()`: стол, кластеры, поза (x, y, yaw), ящики, объекты, штативы |
| brain | разбор задачи, дедлайн, цикл LOOK → PERCEIVE → SELECT → PICK → PLACE → BUDGET, фильтр по `min_confidence`, гарантированный `done` | пробирки, PLAN, приоритет уроненных, отказ по `p · P`, проверка веса в запястье, RESCAN |
| grasp | `plan_grasps` (сверху, 2 кандидата), `plan_place` для ящиков, `grasp_db.yaml` (параметрические шаблоны) | позы захвата в системе объекта, аналитический запасной вариант, успешность по пробам, ориентация TCP (`top_down`) не проверена |

Пока IK и детекция не готовы, решение запускается, ничего не трогает и публикует `done`: 0 баллов без штрафов.

## Как отлаживать модули изолированно

| Модуль | Как | Метрика |
|---|---|---|
| motion | скрипт шлёт в `motion/move_tcp` сетку целей над столом и сверяет с `tf2_echo pelvis right_tcp_link` | ошибка < 1 см, время выхода, доля `unreachable` |
| perception | rosbag кадров и tf на 30 сидах; эталон получаем через `/sim/objects/set`, только в `tools/` | полнота, точность категорий, ошибка xyz |
| grasp и skills | объекты ставим в известные точки через `/sim/objects/set`, perception подменяем заглушкой, которая возвращает эти точки | доля успешных захватов по каждому кандидату, она же успешность кандидата в `grasp_db.yaml` |
| brain | заглушки perception и motion с тем же интерфейсом | проходит все задачи, `done` публикуется |

Заглушки и отладочные скрипты лежат в `tools/`, в `src/` их нет. `tools/check_rules.sh` это проверяет.

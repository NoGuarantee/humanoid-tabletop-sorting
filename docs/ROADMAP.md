# План работ

Владельцев впишите сами. Роли: **motion** (IK, движения, кисти), **perception** (камеры, детекция),
**grasping** (захваты, skills, grasp_db), **brain/infra** (state machine, запуск, проверки, README, скринкасты).
Если людей меньше четырёх, grasping обычно совмещают с motion, а brain/infra с perception.

| Роль | Кто |
|---|---|
| motion | |
| perception | |
| grasping | |
| brain/infra | |

## M0. Скелет (сделано в первом коммите)

- [x] Пакеты `hack_interfaces`, `hack_motion`, `hack_perception`, `hack_brain`, `hack_bringup`
- [x] `run.sh`, launch-файл, гарантированная публикация `/sim/task/done`
- [x] `tools/check_rules.sh`, `tools/dev_shell.sh`
- [x] Проверить, что `bash hackathon/tools/solution.sh ~/…/nssr-sol-team-049` собирает и запускает решение
  (brain/infra). 2026-11-03, `remote.sh eval sort3 --pool 4 --seed 1`: сборка 10 с, три узла поднимаются,
  `/sim/task/done` через 20 с, балл 0 (perception ещё не реализован)
- [x] Проверить nvcc и память GPU в образе (brain/infra). В `nssr-sim:latest` нет nvcc и CUDA toolkit, cuRobo v0.8
  собирает ядра через NVRTC; Isaac занимает около 4.3 из 16 ГБ VRAM; драйвер 580, compute capability 12.0
- [ ] Задать организаторам вопросы из [ANALYSIS.md](ANALYSIS.md), раздел 9 (brain/infra)

## M1. Каждый модуль работает сам по себе

- [x] motion, первый день: первичные проверки cuRobo на сервере ([ANALYSIS.md](ANALYSIS.md), раздел 7.3) — сборка
  в контейнере из образа, память GPU рядом с симулятором, прогрев, время плана; версия V2 v0.8.0. Прошли
- [ ] motion: установка cuRobo в `setup.sh` (сборка wheel как в `tools/curobo_install.sh`) и прогрев кэша Warp там
  же; проверить через полный `eval`. Написано, `setup.sh` работает в `dev`; полного `eval` ещё не было
- [x] motion: конфиг TienKung для cuRobo (сферы коллизий рук, корпуса и головы). 2026-11-03, меши из опубликованного
  URDF производителя, сферы и матрица самоколлизий в `src/hack_motion/robot/` (см. README там же); IK обеих рук
  без мешей 20/20, 3 мс
- [ ] motion: бэкенд `planner: curobo` (конфиг из `share/hack_motion/robot`, `lock_joints` из `/joint_states`,
  подъём рук по суставам `robot.RAISE_STEPS` перед первым планом). Работает, при подъёме силы на запястьях до 9 Н
  (самопересечения это не исключает, траекторию проверяет `tools/trajectory_self_check.py`). Планировщик держит
  `|shoulder_roll|` не больше 1.3: при ±π/2 оси `shoulder_yaw` и `shoulder_pitch` совпадают, и `shoulder_pitch` не
  доходит до цели, лишняя нагрузка до 60 Нм (`tools/shoulder_roll_sweep.py`)
- [ ] motion: MoveTcp с ошибкой TCP < 1 см, проверить ориентацию `top_down` по tf. На дошедших целях 0.2–0.8 см и
  до 2.3°, ось z TCP вверх. `tools/move_tcp_check.py` с ограничением roll: 5 из 6 (все 5 с прямой головой, 0.09–0.39 см
  и до 1.3°) и `--low` 2 из 6.
  Открыто: цель с наклоном головы 0.4 вернула `unreachable` (0/32 сидов) в обычном прогоне и прошла в `--low`; таймауты
  на низких целях с пиком силы 51–107 Н, похоже на касание: стола и объектов в мире cuRobo ещё нет
- [ ] ~~motion, только если cuRobo не прошёл первичные проверки: `ik.solve` без коллизий~~ не нужен, cuRobo прошёл
- [ ] perception: плоскость стола, кластеры, поза (x, y, yaw), размеры, `top_z`
- [ ] grasping: pick и place одного объекта, поставленного в известную точку (заглушка perception в `tools/`).
  Учесть: `plan_grasps` считает z TCP как `max(top_z + z_offset, table_z + 0.015)`, будто TCP на кончиках пальцев,
  а они на 0.21 м ниже TCP, и пальцы уйдут в стол
- [ ] brain/infra: заглушки perception и motion, прогон state machine по всем задачам

## M2. Бейзлайн сортировки, балл > 0 ← главная веха

- [ ] motion: вертикальные подвод, опускание и подъём; стол и ящики в мире cuRobo
- [ ] perception: ящики по цвету, объекты по эвристике размеров и цвета или CLIP zero-shot
- [ ] grasping: параметры `grasp_db.yaml` для пула 4, аналитический запасной вариант по облаку точек
- [ ] brain/infra: полный прогон `sort3 --pool 4` и `sort2 --pool 4`, балл > 0 на 10 сидах

## M3. Надёжность и скорость

- [ ] motion: без лишних возвратов домой; соседние объекты в мире cuRobo, план на набор целей, рука и корпус вместе
- [ ] perception: YOLO-seg по 15 объектам пулов 4 и 6 на кадрах из симулятора, калибровка уверенности; ICP по yaw для
  несимметричных объектов
- [ ] grasping: позы захвата в системе объекта (заготовки из Grasp Editor), пробы кандидатов скриптом в `tools/`,
  успешность в `grasp_db.yaml`, проверка веса в запястье, повторы
- [ ] brain/infra: PLAN и ранжирование кандидатов, приоритет уроненных, отказ при `p · P ≤ 0.5`, RESCAN, бюджет
  времени, прогон на 30+ сидах, пул 6

## M4. Пробирки easy, standard, color

- [ ] motion: поворот корпуса параллельно с движением руки; коллизии со шкафом и штативами (cuRobo)
- [ ] perception: штативы, гнёзда, занятость, крышки, размер пробирок, вид из шкафа
- [ ] grasping: хваты `tube50` и `tube15`, вставка с довыравниванием
- [ ] brain/infra: `run_tubes`

## M5. Лежащие пробирки и сдача

- [ ] perception и grasping: `lying`, примитив перехвата
- [ ] brain/infra: README по каждой задаче, скринкасты в Releases (окно симулятора и строка `layout seed`)

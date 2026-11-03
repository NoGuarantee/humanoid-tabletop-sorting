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
- [ ] Проверить, что `bash hackathon/tools/solution.sh ~/…/nssr-sol-team-049` собирает и запускает решение (brain/infra)
- [ ] Задать организаторам вопросы из [ANALYSIS.md](ANALYSIS.md), раздел 9; проверить nvcc и память GPU в образе (brain/infra)

## M1. Каждый модуль работает сам по себе

- [ ] motion, первый день: первичные проверки cuRobo на сервере ([ANALYSIS.md](ANALYSIS.md), раздел 7.3) — сборка
  через `solution.sh`, память GPU рядом с симулятором, прогрев, время плана; закрепить версию (V2 v0.8.x или v0.7.8)
- [ ] motion: конфиг TienKung для cuRobo (сферы коллизий рук, корпуса и головы), бэкенд `planner: curobo`
- [ ] motion: MoveTcp с ошибкой TCP < 1 см, проверить ориентацию `top_down` по tf
- [ ] motion, только если cuRobo не прошёл первичные проверки: `ik.solve` без коллизий
- [ ] perception: плоскость стола, кластеры, поза (x, y, yaw), размеры, `top_z`
- [ ] grasping: pick и place одного объекта, поставленного в известную точку (заглушка perception в `tools/`)
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

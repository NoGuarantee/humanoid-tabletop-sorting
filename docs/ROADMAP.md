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

## M1. Каждый модуль работает сам по себе

- [ ] motion: `ik.solve`, MoveTcp с ошибкой TCP < 1 см, проверить ориентацию `top_down` по tf
- [ ] perception: плоскость стола, кластеры, центр, yaw, размеры, `top_z`
- [ ] grasping: pick и place одного объекта, поставленного в известную точку (заглушка perception в `tools/`)
- [ ] brain/infra: заглушки perception и motion, прогон state machine по всем задачам

## M2. Бейзлайн сортировки, балл > 0 ← главная веха

- [ ] motion: режим `linear`
- [ ] perception: ящики по цвету, категории (CLIP zero-shot или эвристики)
- [ ] grasping: параметры `grasp_db.yaml` для пула 4
- [ ] brain/infra: полный прогон `sort3 --pool 4` и `sort2 --pool 4`, балл > 0 на 10 сидах

## M3. Надёжность и скорость

- [ ] motion: синхронизация суставов, без лишних возвратов домой
- [ ] perception: YOLO-seg на кадрах из симулятора всех 6 пулов, калибровка уверенности
- [ ] grasping: проверка веса в запястье, повторы, возврат объекта при неудаче
- [ ] brain/infra: бюджет времени, порог `p > 0.5`, прогон на 30+ сидах, пул 6

## M4. Пробирки easy, standard, color

- [ ] motion: поворот корпуса параллельно с движением руки
- [ ] perception: штативы, гнёзда, занятость, крышки, размер пробирок, вид из шкафа
- [ ] grasping: хваты `tube50` и `tube15`, вставка с довыравниванием
- [ ] brain/infra: `run_tubes`

## M5. Лежащие пробирки и сдача

- [ ] perception и grasping: `lying`, примитив перехвата
- [ ] brain/infra: README по каждой задаче, скринкасты в Releases (окно симулятора и строка `layout seed`)

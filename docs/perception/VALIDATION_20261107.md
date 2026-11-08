# PR #5: интеграция и фактическая проверка, 7 ноября 2026

Последующая live-проверка первого ответа опубликована в
[LIVE_PERCEIVE_20261108.md](LIVE_PERCEIVE_20261108.md). Ниже — исторический
снимок кода/проверок до этой доработки.

PR остаётся draft. Live-приёмка не закрыта. Данные ниже — программные проверки,
сохранённый sensor replay и изолированный ROS replay; успешный захват или
официальный балл этим отчётом не подтверждаются.

## Источники и границы diff

- Основа PR: `1b5da205c0a8568cc09eba7fad8a48732fd13ad3`.
- Выбранная sort3-копия: `5d8ac2effd8324b91a796f88db244bdf2107908c`;
  выбранные изменения perception сверены с ней.
- Multi-object-копия: `aeb6732227b711cd3d3478272bd7eacfd2431bec`;
  перенесены perception/backend/service/worker и тесты разделения компонент/
  timing, без brain/motion/grasp-кода и их конфигурации.
- Актуальный `main`: `b69da20832700c0077e92e65d250932f9bb8a177`, включён
  обычным merge `92edfebd5ddb54795af147a179874b4268293b9a`.
  Diff к этому main не меняет brain, motion или hack_interfaces. PR #4 не менялся.
- Собственная чистая копия:
  `/Users/vlad/projects/ros_hack/worktrees/pr5-perception-clean-20261107`.
  Исследовательские копии/ветки и остановленные grasp-кампании сохранены.

Дополнительные исправления интеграции: препятствия выдаются и без фильтра
моделей, и для соседей вне пула; высокие поверхности/малые маски не удаляются
как предполагаемая рука; bounds сохраняют все видимые точки; неизвестные
компоненты имеют отдельные IDs. `exposure_key` повторён на объектах/лотках/
препятствиях. Spawn/join/restart и логи вынесены из input lock. Повторно
проверяется свежесть после decode/TF. Ошибка restart-spawn не рушит сервис.

## Программные проверки

- **38 тестов** на Mac (переиспользованная Python 3.12 среда, NumPy 2.5.4,
  SciPy 1.18.1). В том числе: одна/две камеры, точный stamp/K/TF, компоненты,
  остатки, исключение только primary ID, независимость препятствий от фильтра,
  повторный fit той же экспозиции после ошибки запрещён, reset/expiry,
  shutdown вне lock и отказ нового spawn.
- **38 тестов**, пять пакетов `colcon build`, реальные ROS Humble imports,
  launch `--show-args` — в отдельном CPU-контейнере `--network none`, 2 CPU,
  4 GiB RAM, без GPU. Это сборка dev-image, **не official solution.sh**.
  [Полный build/test log](validation_20261107/build-final.log).
- `check_rules.sh`, `git diff --check`, Python compileall и shell syntax прошли.
- Реальный архив SHA256 `b33b2debe3c1710bd6e7458b983767bfc7a7ce003b0d9c5560ad20ec5bd7fade`
  извлечён дважды; повреждённый архив отвергнут до публикации каталога.
  Маршрутизация `run.sh` без/с marker проверена с заглушкой `ros2`; это только
  shell-проверка выбора режима, не ROS запуск.

## Производительность сохранённых кадров

4090, Torch 2.8.0+cu126, 2 Torch CPU-потока, OPENBLAS_NUM_THREADS=1. По три
сохранённых RGB-D/CameraInfo/TF пары, 4 повтора, медиана последних трёх.
Входы без native physics/GT/seed координат; точные входные и кодовые hashes
находятся в JSON. Время не включает ROS/DDS, IPC worker и ожидание нового кадра.

| Сохранённая пара | Итоговый CPU, все модели | CUDA snapshot, все модели | Итоговый CPU, фильтр rhino/Nescafe | Регионов, все / фильтр |
|---|---:|---:|---:|---:|
| native-pilot…000000 | 2,541 с | 2,467 с | 0,678 с | 14 / 14 |
| sink…000082 | 1,632 с | 1,546 с | 0,688 с | 11 / 11 |
| mouse…000060 | 2,565 с | 2,498 с | 0,773 с | 19 / 19 |

Итоговый CPU и фильтрованный CPU совпадают по SHA256 **каждого production .py**
с опубликованной реализацией. CUDA snapshot сделан до двух последних изменений:
нового статуса `not_fitted` для нефитированных масок при фильтре и обработки
ошибки повторного spawn; численный путь полного fit в нём тот же. Повтор
CUDA на окончательном дереве **не выполнен**: общий GPU lock был занят.
Исторические цифры 2,56–2,76× YOLO из старого sort3-benchmark не выдаются за
измерение этой реализации. В новом CUDA snapshot YOLO занимает 36–45 мс
для двух камер, полный pipeline — 1,55–2,50 с. CPU ICP остаётся узким местом.

Пиковый RSS отдельных benchmark-процессов: CPU 905 MiB, CUDA snapshot
1695 MiB. Пик **выделенной Torch CUDA памяти** 75 697 664 байта (72,2 MiB),
это не полный VRAM процесса/драйвера. Эти цифры не измеряют совместную память
с simulator/cuRobo. На данных трёх парах максимальное расхождение принятых
CPU/CUDA поз 0,115 мм / 0,103°; это согласованность устройств, не точность
или обобщение на новые раскладки. Полный fit при ходе sim-time 1× может
протухнуть по TTL 1,75 с. Ускорение фильтром требует заранее выбранного
списка, не доказывает ускорение всего цикла сортировки.

[Полный CPU](validation_20261107/benchmark-cpu.json),
[CUDA snapshot](validation_20261107/benchmark-cuda-0.json),
[фильтрованный CPU](validation_20261107/benchmark-filtered-cpu.json),
[hash manifest и сравнение поз](validation_20261107/manifest.json).
В raw receipts могут быть дополнительные `._*` hashes от метаданных первого
Mac-архива; они не являются Python-модулями. Все production hashes сверены.

Воспроизведение в совместимой среде, отдельно для каждого устройства
(Ultralytics CPU selection меняет CUDA visibility внутри процесса):

```bash
PYTHONPATH=src/hack_perception OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python3 tools/perception_rgbd_benchmark.py --data /path/to/saved_sensor_packet \
  --bundle weights/rgbd-b33b2debe3c1 --device cpu --output cpu.json
# Другой процесс; на общем GPU сначала cooperative lock:
PYTHONPATH=src/hack_perception OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python3 tools/perception_rgbd_benchmark.py --data /path/to/saved_sensor_packet \
  --bundle weights/rgbd-b33b2debe3c1 --device cuda:0 --output cuda.json
# Для фильтра добавить --names Schleich_African_Black_Rhino,Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g
```

## Изолированный настоящий ROS replay

В CPU-only контейнере, `--network none`, domain 209, 4 CPU / 6 GiB,
публиковалась сохранённая v35 RGB-D сцена. TF камер — из того же сенсорного
снимка; повторяемым изображениям задавались текущие wall-clock stamps. TF рук
не публиковались. Это проверка ROS callbacks/IPC, не новый физический эпизод.
Использовался код до тех же двух ненумерических изменений, указанных выше.

- Получены принятые однокамерные позы Nescafe и rhino с единым exposure_key
  у поз и всех наблюдаемых регионов. Модель работала на CPU.
- Два обычных `Perceive` RPC заняли **1,397 и 1,365 с** и оба корректно
  **отклонены по TTL**, а не выданы свежими. Сервис доступен, но бюджет
  latency/freshness не пройден. Не было достаточного числа наблюдений с
  active pose job, чтобы объявить доказанным SLA `Perceive` во время 6D.
- После реального `terminate()` worker следующий RPC завершился за 1,282 с
  безопасным `exposure invalidated`; зафиксирован один restart. Успешный
  свежий fallback после восстановления этим коротким опытом не подтверждён.
- Отсутствующий TF рук вызывает штатные ожидания/предупреждения; причинный
  вклад TF, кластеризации, CPU и логов в полную задержку не изолирован.
  Нельзя переносить этот результат на live стенд как latency guarantee.

[Сырой ROS ответ](validation_20261107/ros-replay.json),
[точный probe](validation_20261107/ros_replay_probe.py).
Его exit code 0 означает, что probe завершился и сохранил данные; это не pass
критерия свежего Perceive. Он предназначен только для изолированной сети.

## Checklist актуального review PR #5

| Пункт | Фактический статус |
|---|---|
| Первый свежий ответ сразу после LOOK_TABLE, Sort3/pool 4 | **Открыт**, live не запускался; replay freshness отрицательный |
| Perceive во время полной 6D | **Открыт** по live SLA; процесс изолирован, короткий ROS replay бюджет не прошёл |
| Нет весов / падение worker | Unit fallback и реальное terminate/restart проверены частично; успешный свежий live fallback **открыт** |
| Timestamp/TF/одна камера/один fit | Контрактные тесты + реальная однокамерная 6D в replay; новая live DDS приёмка **открыта** |
| Память рядом с cuRobo/simulator | **Открыт**; только standalone benchmark RSS/Torch allocation |
| Occlusion кистью | Сохранённый v35 кадр с руками и тесты разрыва масок; live полнота препятствий/ошибки pose **открыты** |
| Official solution.sh | **Не запускался**; есть изолированная dev сборка/ROS replay |
| Фактические задержки | Измерены этапы replay и отрицательные ROS RPC; live end-to-end **открыт** |
| Допустимость offline training/shape bundle | **Решение команды отсутствует**, блокер сдачи 6D |
| Grasp и официальный score | Не проверялись, не являются результатом этого PR |

Официальный solution.sh жёстко использует network host/domain 146. На общем
сервере работают другие контейнеры/исследования, поэтому новый simulator и
полное решение с управляющими узлами не запускались. На 4090 использовался
общий `gpu0.lock`; занятый lock не обходился. Новые grasp-кампании не запускались.

Удалённые собственные каталоги:
`nssr-rtx6000:/home/spiridonov/vlad/junk/nssr-uwt-20261030/pr5-perception-integration-20261107`
и `vast4090:/root/vlad/junk/nssr-uwt-20261030/pr5-perception-integration-20261107`.
Переиспользован image `nssr-dev:6057b5989a06`, digest
`sha256:b3cfb38b84c65a7947c7f6475010b08ac9c61dd9bd667afe86bfbc5ae2933121`.
В ROS replay только Ultralytics/TorchVision взяты из остановленного собственного
sort3-контейнера в отдельный runtime overlay; исходный контейнер не менялся.
Собственные bounded jobs закончены, тестовые контейнеры удалены штатным
`--rm`; чужие процессы, симуляторы и среды не трогались. Компактные результаты
фактически возвращены на Mac в `outputs/pr5_perception_integration_20261107/`.

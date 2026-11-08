# RGB-D сцена и модельные позы через ROS

PR #5 добавляет опциональный RGB-D worker к штатному `/perception/perceive` и
`/perception/estimate_model_poses` (`std_srvs/Trigger`). `SceneObjects`,
`hack_interfaces`, brain и motion сохраняют свои интерфейсы. Код захватов и
выбор экспозиции в grasp-клиенте в этот PR не входят.

Онлайн-входы геометрии — RGB, зарегистрированная глубина в мм, CameraInfo и TF
на время экспозиции. `/sim/task` используется для пула/соответствия цвета и ID
лотка и смены эпохи; `/sim/objects/names` — для штатных категорий. Ни seed
раскладки, ни координаты USD/physics/GT, ни `/sim/task/check`, ни debug setters
решению не нужны. Фиксированный random seed алгоритма RANSAC/ICP не является
seed раскладки. Сохранённые сенсорные кадры используются только для replay.

## Поставка и включение

Без пакета запускается штатный детектор. Подготовьте существующий приватный
[model bundle](https://github.com/nssr-hackathon/nssr-sol-team-049/releases/tag/pr2-rgbd-model-20261105)
(около 12,2 МБ) до запуска evaluator:

```bash
mkdir -p weights
gh release download pr2-rgbd-model-20261105 -R nssr-hackathon/nssr-sol-team-049 \
  --pattern geometry_only_ros_bundle.tar.gz --dir weights
bash tools/stage_perception_rgbd.sh
```

SHA256 архива:
`b33b2debe3c1710bd6e7458b983767bfc7a7ce003b0d9c5560ad20ec5bd7fade`.
Скрипт публикует checksum-addressed каталог `weights/rgbd-b33b2debe3c1`.
Backend проверяет хеши весов и каталога, а 6D также проверяет каждое облако.
Архив доставляется вместе с выбранным исходным деревом; `remote.sh sync`
исключает `weights/`, его передают отдельно. Никаких загрузок весов из runtime.

`setup.sh` независимо от успеха cuRobo подготавливает имеющийся архив,
устанавливает pinned зависимости из `tools/perception_rgbd_requirements.txt`
после evaluator PyTorch 2.8.0 и проверяет NumPy 1.x. Только успешная подготовка
создаёт `weights/.rgbd-ready`; `run.sh` тогда задаёт каталог модели. Без архива
ничего не скачивается. Ultralytics offline/autoinstall выключены в worker.

```bash
# В ROS Humble среде после colcon build и source install/setup.bash:
bash run.sh rgbd_enable_model_poses:=true rgbd_device:=cuda:0 \
  rgbd_on_demand:=true \
  rgbd_model_names:=Schleich_African_Black_Rhino,Nescafe_Memento_Latte_Caramel_8_08_oz_23_g_packets_64_oz_184_g
```

| Launch / node parameter | Environment default | Значение по умолчанию |
|---|---|---|
| `serialized_images` | нет | true: XCDR1 raw Image; false: typed ROS Image |
| `perceive_frame_wait_s` | нет | 0,75 wall s; bounds 0..1 |
| `rgbd_model_dir` | `NSSR_RGBD_MODEL_DIR` / verified marker | пусто |
| `rgbd_enable_model_poses` | `NSSR_RGBD_ENABLE_MODEL_POSES` | false |
| `rgbd_device` | `NSSR_RGBD_DEVICE` | cpu |
| `rgbd_on_demand` | `NSSR_RGBD_ON_DEMAND` | false |
| `rgbd_model_names` | `NSSR_RGBD_MODEL_NAMES` | пусто: все модели пула |
| `rgbd_torch_threads` | `NSSR_RGBD_TORCH_THREADS` | 2, допустимо 1–4 |
| `rgbd_pool` | нет | 0: из задачи; override 4/6 |

`rgbd_model_names` — read-only параметр старта, имена через запятую с удалением
пробелов. Он отсекает неподходящие категории до подготовки pose tasks и
гипотезы до дорогого ICP; соседние маски/препятствия не фильтруются. CUDA
использует YOLO; learned identity/pose heads и depth/ICP остаются на CPU.
Запрошенная недоступная CUDA даёт явную ошибку worker и штатный fallback.
Параметры perception не зависят от режима brain/grasp.

## Планирование и свежесть

Одна ROS-подписка на поток: reliable KEEP_LAST=2, как у моста. Оба CameraInfo
подписаны также без model bundle; baseline и worker используют один буфер целых экспозиций. Внутри камеры нужны одинаковые stamp, optical frame,
разрешение и pinhole K; две камеры объединяются только при общем stamp.
Иначе допускается одна полная камера. TF запрашивается на время кадра.

Worker — отдельный spawn-процесс, одна задача и одна подготовленная экспозиция.
Геометрия запускается не чаще 3 раз/с, вход не старше 0,35 sim s. Маски и
плоскость переиспользуются для 6D, не вычисляются повторно. Каждая экспозиция
ставится на 6D не больше одного раза, включая неудачный fit.

`Perceive` ждёт целую экспозицию не раньше начала запроса и не позже текущего
ROS clock, до 0,75 wall s (параметр `perceive_frame_wait_s`), отпуская input lock.
Если такая не пришла, отдаётся последняя целая экспозиция любого возраста, `success`
true и `message` начинается со `stale frame`: пустой ответ останавливает brain.
TF камеры ждётся по `/clock` до 2 sim s, рук до 0,2 sim s; если TF так и отстаёт,
берётся последний, когда звено не сдвинулось за 0,5 s до него (`lookup_at`).
Он не ожидает inference worker: готовая сцена подставляется только до **0,65 sim s**
с исходными RGB-D того же stamp. Иначе идёт штатный sensor-based путь.
Предел **1,0 sim s** — отказ до/после обработки `Perceive`, а не предел
подстановки worker-сцены. Высота стола относится к `(0.38, 0)` в pelvis.
Недостающие ID лотков добираются штатным цветовым детектором, даже при пустом
списке модельных лотков.

6D opt-in: первый Trigger ставит запрос, пока нет результата возвращается
`awaiting_fresh_pose`. Вход fit не старше 0,65 sim s; готовый ответ не старше
1,75 sim s. В continuous режиме кэш возвращается повторно до TTL. В
`rgbd_on_demand=true` worker простаивает без запроса; следующий цикл требует
экспозицию строго новее момента запроса, успешный результат потребляется
один раз. Повторный polling активного fit не ставит второй fit. Этот режим
имеет смысл вместе с включённым сервисом 6D; без запросов `Perceive` использует
штатный путь. TTL не продлевается при выдаче ответа. Полный CPU ICP нередко
превышает TTL при симуляции 1×; фильтр моделей уменьшает объём работы, но не
гарантирует свежий ответ для любой сцены.

Clock rollback/смена clock и task reset очищают оба буфера, трекер и результаты;
при clock reset очищается TF. Результаты старой эпохи отвергаются. Worker
ограничен 25 с загрузки и 20 с на задачу, допускаются два перезапуска.
Shutdown/join/spawn и диагностическое логирование идут вне общего input lock.
Executor ограничен четырьмя потоками, camera/pump/Trigger используют отдельные
callback groups. Диагностика `rgbd_rpc_timing`, `rgbd_pump_timing`,
`rgbd_worker_timing`, `perceive_timing` показывает ожидание lock, стадии
pump/decode/TF, возраст кадров и pending job. Она не сериализует изображения
или геометрию; время до входа в callback измеряется внешним RPC-клиентом.

## JSON и ID экспозиции

Сохраняется `hack_perception.model_poses/v1`, `stamp_ns`, `frame_id="pelvis"`,
`objects`, `containers`, `table_z_m`. `Trigger.success` означает обработанную
экспозицию; пригодность каждого объекта определяется `pose_status`.
`T_pelvis_object` и `center_pelvis_m` есть только при `ok`.

Дополнения: `table_normal_pelvis`, `T_pelvis_opening` у лотка (оси X/Y в
измеренной плоскости, Z — нормаль, origin — центр проёма),
`observed_regions`, `source_cameras`, `exposure_key`. Последний содержит
`{session_id, epoch, stamp_ns}` и повторён у каждой позы, лотка и региона.
`session_id` меняется при новом worker, `epoch` — при reset. Потребитель
сравнивает **весь exposure_key и camera/instance ID**, прежде чем исключать
целевой регион; один integer instance ID между кадрами сравнивать нельзя.
Вне ROS worker offline `infer_response` возвращает только stamp; ключ
добавляется worker после успешного вычисления. Старые клиенты могут
игнорировать новые поля.

`observed_regions` всегда выдаются, независимо от фильтра имён и пула. Это
min/max видимых depth-точек масок предметов выше плоскости на 6 мм, а не полный
объём или гарантия свободного пространства. Маски лотков/rack имеют отдельную
геометрию и сюда не входят. Confidence и высота сами по себе не доказывают,
что поверхность принадлежит руке: высокие и слабые маски сохраняются.
Связность: voxel 3 мм, соседство 12 мм; основной компонент наследует ID маски,
остальные поверхности и объединённые мелкие остатки получают отдельные
отрицательные IDs и `category: unknown`. `source_instances` у них — только
происхождение; исключать их как целевой объект нельзя. Даже маска из <30
точек остаётся препятствием. Несвязные остатки могут дать консервативный
большой box. Невидимые/не сегментированные поверхности не восстановлены;
соприкасающаяся рука может остаться в основном компоненте.

| `pose_status` | Значение |
|---|---|
| `ok` | ≥1 целого depth-ракурса и пригодная подгонка; число в `depth_supported_views` |
| `reacquire_clipped` | Все поддерживающие маски касаются края кадра |
| `reacquire_uncertain_fit` | Невалидная матрица, residual >12 мм или ненадёжная ось |
| `insufficient_depth` | Маска без пригодной depth-подгонки в режиме всех моделей |
| `not_fitted` | Маска без fit в режиме ограничения имён; причина может быть фильтр или глубина |

## Происхождение и проверки

Артефакты из исследовательского PR #2 (`a3afb9b`): YOLO сегментация по native
RGB-D train, learned identity/pose/tube/box heads и модельные облака формы из
train с офлайн-разметкой. Это модельные priors, не онлайн-измерение раскладки.
Онлайн-поза заново оценивается по кадру. Допустимость этих обучающих данных
для сдачи **не согласована**; решение команды остаётся открытым блокером 6D.
Полная история обучения/данных находится в linked research PR/release.

Живая проверка первого ответа: **6/6** свежих после LOOK_TABLE, RPC 0,324–0,593 s,
возраст 0,23–0,45 sim s. Полный контракт, версии и оставшиеся ограничения:
[LIVE_PERCEIVE_20261108.md](LIVE_PERCEIVE_20261108.md).
Предыдущие offline/replay измерения остаются историческим снимком:
[VALIDATION_20261107.md](VALIDATION_20261107.md).
На отдельном live Sort3/pool 4 после штатного `LOOK_TABLE`:

```bash
python3 tools/perception_model_pose_check.py --scene-only --require-first-tray
python3 tools/perception_model_pose_check.py --require-first-tray --require-pose
ros2 topic info /cam_head/color/image_raw --verbose
```

Повторить с выключенной моделью, недоступными весами, реальным падением worker,
при перекрытии кистью и рядом с загруженным cuRobo. Проверить official
`hackathon/tools/solution.sh`, первый ответ и ROS/DDS latency. Replay/моки не
заменяют эту приёмку. Готовность к слиянию и захват этим PR не подтверждены.

# TienKung 2.0 Pro: модель для motion

| Файл | Что это |
|---|---|
| `tiangong2pro_with_hands.urdf` | URDF `tiangong2pro_urdf/urdf/tiangong2.0_pro_with_hands.urdf` из [Open-X-Humanoid/TienKung_URDF](https://github.com/Open-X-Humanoid/TienKung_URDF) на коммите `4c2c061d72dc`, без изменений. Совпадает с URDF симулятора, отличаются только комментарии |
| `tienkung_curobo.yml` | Конфиг робота для cuRobo v0.8: сферы коллизий, матрица самоколлизий, `lock_joints`. Сгенерирован, руками не правим |
| `LICENSE.TienKung_URDF` | OpenAtom Open Hardware License 1.0, под ней опубликованы URDF и меши |

Сферы в `tienkung_curobo.yml` подобраны по STL-мешам того же коммита. Это производные данные, а не исходные
файлы (требование лицензии, п. 3.1: изменения отмечены). Самих мешей в репозитории нет, в рантайме они не нужны.

Как пересобрать на сервере (меши кэшируются в `data/`, сверяются с хешами git):

```bash
bash tools/remote.sh dev --timeout 3600 -- 'bash tools/curobo_install.sh && python3 tools/fit_tienkung_spheres.py'
bash tools/remote.sh fetch work/$NSSR_WHO/nssr-sol-team-049/src/hack_motion/robot .fetch   # до следующего sync
cp .fetch/robot/tienkung_curobo.yml src/hack_motion/robot/
```

## Как motion загружает конфиг

1. Читает yml и ставит `kinematics.urdf_path` на установленную копию URDF в `share/hack_motion/robot/`.
   `asset_root_path` пустой, мешей нет.
2. Подменяет значения в `lock_joints` для ног и головы на текущие из `/joint_states`. В yml записана домашняя
   поза симулятора: `hip_pitch` -0.5, `knee_pitch` 1.0, остальное 0.
3. Отдаёт словарь в `InverseKinematicsCfg.create(robot=...)` и `MotionPlannerCfg.create(robot=...)`.
4. Перед первым планированием переводит руки по суставам (без планировщика) в позу по умолчанию
   `cspace.default_joint_position`. Это домашняя поза, в которой плечи отведены от корпуса на 0.2 рад
   (`shoulder_roll`). В домашней позе сферы предплечий заходят в бёдра на несколько мм. Если бы матрица
   самоколлизий строилась в ней, пара «предплечье–бедро» попала бы в игнор, и планировщик мог бы провести
   предплечье сквозь бедро. Поэтому матрица построена в отведённой позе, а старт из домашней позы планировщик
   считает столкновением.

Активные суставы: `body_yaw_joint` и по 7 суставов на руку. Инструменты: `left_tcp_link`, `right_tcp_link`.
Пальцев в дереве cuRobo нет: их сферы в открытом положении перенесены на ладони (`L_base_link`, `R_base_link`).
Кистью управляют пресеты, а не планировщик. Есть и вторая причина: cuRobo понимает только оси суставов вдоль x/y/z,
а у `right_index_1_joint` в опубликованном URDF ось наклонена на 4°.

# model_server

Self-contained ROS 2 (ament_python) package hosting the project's heavy
model-inference servers. Each server is a single file that imports nothing from
other workspace perception packages; the servers share only this package's
`model_logging` module (which `golem_skills` imports too).

## Executables

| Executable | Service | Interface | Notes |
|---|---|---|---|
| `gemini_server` | `/gemini_query` | `custom_ros_messages/srv/GeminiQuery` | Image/text → Gemini text. Needs `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). |
| `sam_server` | `/sam_segment` | `custom_ros_messages/srv/SamSegment` | SAM3 promptable segmentation. Loads `weights/sam3.pt`. |
| `graspgen_server` | `/graspgen` | `custom_ros_messages/srv/GraspGen` | GraspGenX 6-DOF grasp planning. Loads `weights/graspgen/release/{gen,dis}`. |
| `yolo_server` | pub/sub: `<image_topic>/detections` | `custom_ros_messages/msg/DetectionBundle` | YOLO-World open-vocabulary detection. Subscribes to N `image_topics` (CompressedImage), publishes a DetectionBundle per input topic at `publish_rate_hz` (default 5 Hz). Classes come from the live `queries` param. Loads `weights/yolo_world_battery_best.pt`. |

Run with `ros2 run model_server <executable>`.

## Weights

All checkpoints live under `weights/`. The YOLO-World and GraspGenX ones are
tracked with Git LFS; `sam3.pt` is not in git (`weights/.gitignore` ignores it):
download it yourself (gated, see `weights/README.md`):

```
weights/
  sam3.pt                                    # SAM3 (facebook/sam3)
  yolo_world_battery_best.pt                 # YOLO-World, fine-tuned for the battery workcell
  graspgen/release/gen/{config.yaml,epoch_736.pth}   # GraspGenX generator
  graspgen/release/dis/{config.yaml,epoch_1056.pth}  # GraspGenX discriminator
```

`sam_server` and `yolo_server` resolve weights relative to this package via
`__file__` (works with `colcon build --symlink-install`); `MODEL_SERVER_WEIGHTS_DIR`
overrides that root for those two. `graspgen_server` reads its checkpoints only
from `GRASPGENX_CHECKPOINT_DIR` (required; the ros compose sets it to this
package's `weights/graspgen`) and its gripper description from
`GRASPGENX_ASSETS_DIR` (required; the ros compose sets it to
`/opt/graspgenx/assets`, where the ros image stages it).

## Dependencies

ROS deps are in `package.xml`. The model libraries are pip-installed in the ros
container and have no rosdep keys: `google-genai`, `pillow` (gemini); `torch`,
`pillow`, `sam3` (sam); `torch`, `graspgenx` (graspgen); `torch`, `ultralytics`
(yolo).

traffic-vision-service/
├── configs/
│   └── cameras/
│       ├── cam_01.yaml         # <-- Put your camera calibration here
│       └── cam_02.yaml
├── api/
│   └── routes/
│       └── jobs.py
├── pipeline/
│   ├── engine.py
│   ├── tracker.py
│   ├── traffic_light.py
│   └── rules.py
├── storage/
│   └── uploads/
├── tools/
│   └── calibrate.py
├── main.py
└── requirements.txt
# imagetranslation-keras


### How to get started

```bash
git clone https://github.com/tbullmann/imagetranslation-keras.git
cd imagetranslation-keras
conda create env tfCPU Python=3.6
activate tfCPU
pip install -r requirements.txt
makedir datasets
cd datasets
git clone https://github.com/tbullmann/groundtruth-drosophila-vnc.git vnc
```



### Folders
Folders structure and important files:
```
.
├── datasets
├── networks
├── utils
│   ├── __init__.py
│   ├── callbacks.py
│   └── ...
├── LICENCE.md
├── README.md
├── translate.py
└── ...
```


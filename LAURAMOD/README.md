# LAURAMOD

This is a modularized version of the `LAURA.py` script. The original script has been broken down into a more organized and maintainable structure.

## Directory Structure

```
LAURAMOD/
├── config/
│   ├── config.py
│   └── __init__.py
├── data/
│   ├── dataset.py
│   ├── transforms.py
│   └── __init__.py
├── engine/
│   ├── evaluator.py
│   ├── trainer.py
│   └── __init__.py
├── models/
│   ├── blocks.py
│   ├── model.py
│   └── __init__.py
├── utils/
│   ├── logger.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── utils.py
│   └── __init__.py
└── main.py
```

## How to Run

To run the training, simply execute the `main.py` script from within the `LAURAMOD` directory:

```bash
python main.py
```

All configurations can be adjusted in `config/config.py`.

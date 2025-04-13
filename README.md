
## Whole Sclice Images Features Classification

### Environment

- python 3.11
- pip 23.3.1
- torch 2.5.1
- pytorch-lightning 1.7.7

```bash
pip install -r requirements.txt
```

### Train
```bash
python main.py --stage train --config path/to/config --batch_size 32
```

Resume checkpoint

```bash
python main.py --stage train --config path/to/config --batch_size 32 --resume path/to/model
```

### Test
```bash
python main.py --stage test --config path/to/config --resume path/to/model
```

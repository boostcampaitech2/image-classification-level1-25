# 슬기로운 캠퍼생활

### 프로젝트 구조
```
image-classification-level1-25/
├── datasets/ *dataset.py
├── models/ *model.py
├── module/
├── save/
├── trans/ *trans.py
│
├── *train.py
├── *inference.py
├── *teameval.py
│
├── TTA_inference.py
├── TTA_teameval.py
│
├── origin_trainer.ipynb
│
└── evaluation.py
```
- '*' `default code`
- 각 폴더내에 실험용 code를 별도의 파일로 작성하여도 `train.py`에서 인자로 불러와 사용할수 있도록 구성하여 `master branch`의 간섭을 최소화 함
- `module/` 학습과정 외에 부가적으로 필요한 기능들을 모아둔 폴더
- `teameval.py` 자체 검증용으로 미리 분리해 둔 데이터를 바탕으로 자체 평가를 하여 실제 `Competition`에 제출하기 전에 검증용으로 활용함
- `TTA_*.py` TTA를 통한 추론/검증 파일로 시간 문제로 `default code`에 통합되지 못했음
- `train_cutmix.py` cutmix를 활용한 학습 코드로 시간 문제로 `default code`에 통합되지 못했음
- `origin_trainer.ipynb` 초기에 쉬운 접근이 가능하도록 작성한 `jupyter notebook` 파일로 후반에는 상호작용이 필요한 작업에 활용됨
- `evaluation.py` 기본으로 제공된 코드이나 채점용 파일로 추측되고 `Competition`참가자가 사용할 필요는 없어보임
- `wandb` 접속을 위한 `AUTH`파일이 필요할 수 있음

### 명령어 샘플
- train
```
python train.py --name isgood --epoch 20 --model rexnet_200base --trainaug A_random_trans --criterion f1 --cutmix False
python train.py --name LambdaLR_cutmix --wandb_unique_tag LambdaLR_cutmix --trainaug A_cutmix_trans2 --epoch 10 --batch_size 64 --mode ALL --val_ratio 0.1 --cutmix True
```
- inference
```
python inference.py --save_dir ./save/test1 --model rexnet_200base
```

### 성능
- 단일모델 기준 `rexnet_200base_corss_4epoch_sota_0.7249`가 public leaderboard 기준 F1 0.7249를 기록하여 1위
- 위 모델에 softmax TTA를 적용한 모델이 private 성능이 가장 잘나왔는데 예상치 못한 결과라 save를 따로 저장하지 못하였음
- 자세한 정보는 `report` 및 `save/` 디렉토리 내의 `.log`나 `.config` 파일 참조

### Requirements
- python, numpy, pandas
- pytorch
- torchvision
- albumentations
- seaborn

import os
import wandb
import argparse
from dotenv import load_dotenv

load_dotenv(dotenv_path='/opt/ml/image-classification-level1-25/wandb.env')

WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')

wandb.login(key=WANDB_AUTH_KEY)


parser = argparse.ArgumentParser()
args = parser.parse_args()

wandb.config.update(args)
# wandb.config.update({"epochs": 4, "batch_size": 32})
wandb.log({
        "Test Accuracy": 100. * 1 / 100,
        "Test Loss": 1})
# show image
wandb.log({"examples": [wandb.Image(numpy_array_or_pil, caption="Label")]})


# read_result.csv
# 결과 데이터 라벨링 분포
wandb.run.summary.update({"gradients": wandb.Histogram(np_histogram=np.histogram(data))})

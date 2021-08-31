import os
import wandb
from dotenv import load_dotenv

def login_wandb(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

def init_wandb(epoch, phase, args, **kwargs):
    tags = [f'unique_tag_name: {args.wandb_unique_tag}',f'epoch: {epoch}', f'user_model: {args.usermodel}', f'userdataset: {args.userdataset}', f'lr: {args.lr}', f'optimizer: {args.optimizer}',f'usertrans: {args.usertrans}']
    name = f'{args.usermodel}, {phase}, {epoch} epoch, {args.wandb_unique_tag}'
    if kwargs:
        for k,v in kwargs.items():
            tags.append(f'{k}: {v}')
            name += f' {v}{k} '
    wandb.init(tags=tags, entity=args.wandb_entity, project=args.wandb_project, reinit=True)
    wandb.run.name = name
    wandb.config.update(args)
    wandb.config.update({"PHASE": phase})


def log_wandb(phase='train', acc=0, f1_score=0, loss=0):
    log = {f"{phase}_acc": acc, f'{phase}_macro_f1': f1_score}
    if phase!='team_eval':
        log['loss'] = loss
    wandb.log({f"{phase}_acc": acc, f"{phase}_loss":loss, f'{phase}_macro_f1': f1_score})

def show_images_wandb(images, preds, y_labels):
    for i in range(len(preds)):
        im = images[i,:,:,:]
        im = im.permute(1,2,0).cuda().cpu().detach().numpy()
        wandb.log({"image_preds": [wandb.Image(im, caption=f"real: {y_labels[i]}, predict: {preds[i]}")]})
        # my_table = wandb.Table()
        # my_table.add_column("image", wandb.Image(im))
        # my_table.add_column("label", y_labels)
        # my_table.add_column("class_prediction", preds)
        # wandb.log({"image_preds_table": my_table})

def draw_result_chart_wandb(y_data):
    y_data = list(y_data)
    y_data_counts = []
    y_data_dict = {x:y_data.count(x) for x in y_data}
    for k,v in y_data_dict.items():
        y_data_counts.append([k,v])
    y_data_table = wandb.Table(data=y_data_counts, columns=["label", "counts"])
    
    wandb.log({"Comparison of results" : wandb.plot.bar(y_data_table, "label", "counts",
                                title="Comparison of results")})

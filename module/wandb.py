import os
import wandb
from dotenv import load_dotenv

def login_wandb(dotenv_path: str):
    '''
    환경변수에 저장된 auth key값으로 wandb에 로그인함
    '''
    load_dotenv(dotenv_path=dotenv_path)
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

def init_wandb(phase, args, **kwargs):
    '''
    kwargs에 fold같은 몇번째 fold인지에 대한 데이터도 줄 수 있음
    
    :param phase: 'train' or 'valid' or 'team_eval'
    :param args: arguments변수
    :param **kwargs: wandb의 태그와 name에 추가하고싶은 내용을 넣어줌 ex_ fold=1
    '''
    tags = [f'unique_tag_name: {args.wandb_unique_tag}', f'{phase}']
    name = f'{phase}, {args.wandb_unique_tag}'

    if phase!='team_eval':
        tags.extend([f'user_model: {args.usermodel}', f'userdataset: {args.userdataset}', f'lr: {args.lr}', f'optimizer: {args.optimizer}',f'usertrans: {args.usertrans}'])
        name = f'{args.usermodel}, {phase}, {args.wandb_unique_tag}'
    if kwargs:
        for k,v in kwargs.items():
            tags.append(f'{k}: {v}')
            name += f' {v}{k} '
    wandb.init(tags=tags, entity=args.wandb_entity, project=args.wandb_project)
    wandb.run.name = name
    wandb.config.update(args)
    wandb.config.update({"PHASE": phase})


def log_wandb(phase='train', acc=0, f1_score=0, loss=0, single_table=False):
    '''
    wandb에 차트 그래프를 그리기 위해 로그를 찍는 함수

    :param phase: 'train' or 'valid' or 'team_eval'
    :paramacc: accuracy
    :param f1_score: f1_score
    :param loss: loss
    :param sing_table: True로 체크하면 train과 validation이 한개의 차트에 표시됨 
    '''
    log = {f"{phase}_acc": acc, f'{phase}_f1_score': f1_score}
    if phase!='team_eval':
        log[f'{phase}_loss'] = loss
    
    if single_table==True:
        log = {acc: acc, 'f1_score': f1_score, 'loss':loss}
    wandb.log(log)

        
def show_images_wandb(images, y_labels, preds):
    '''
    wandb에 media로 이미지를 출력함
    
    :param images: image array를 받음 [batch,channel,width,height]
    :param y_labels: 실제 라벨 데이터
    :param preds: 예측한 데이터 
    '''
    for i in range(len(y_labels)):
        im = images[i,:,:,:]
        im = im.permute(1,2,0).cuda().cpu().detach().numpy()
        wandb.log({"image_preds": [wandb.Image(im, caption=f"real: {y_labels[i]}, predict: {preds[i]}")]})
        # my_table = wandb.Table()
        # my_table.add_column("image", wandb.Image(im))
        # my_table.add_column("label", y_labels)
        # my_table.add_column("class_prediction", preds)
        # wandb.log({"image_preds_table": my_table})

def draw_result_chart_wandb(y_data, title):
    '''
    라벨(x)과 각 라벨별 count(y)를 계산해서 차트에 표시 

    :param y_data: 결과 라벨값 ex_[1,2,3,4,5,1,2,3,1...]
    :param title: 차트에 사용될 제목
    '''
    y_data = list(y_data)
    y_data_counts = []
    y_data_dict = {x:y_data.count(x) for x in y_data}
    for k,v in y_data_dict.items():
        y_data_counts.append([k,v])
    y_data_table = wandb.Table(data=y_data_counts, columns=["label", "counts"])
    
    wandb.log({title: wandb.plot.bar(y_data_table, "label", "counts",
                                title=title)})

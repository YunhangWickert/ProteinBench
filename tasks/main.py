import datetime
import os
import random
import sys; sys.path.append("/liuyunfan/qianyunhang/PFMBench")
sys.path.append('/liuyunfan/qianyunhang/PFMBench/model_zoom')
# sys.path.append(os.getcwd())
# os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0" # gzy
#os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0" # wh
os.environ["WANDB_API_KEY"] = "d4aae42367c9842a7ddfdf29258565305fdd5496" # qyh
import warnings
warnings.filterwarnings("ignore")
import argparse
import pandas as pd
import torch
from pytorch_lightning.trainer import Trainer
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from model_interface import MInterface
from data_interface import DInterface
from model_interface_st2s import MInterface_st2s
from data_interface_st2s import DInterface_st2s
import pytorch_lightning.loggers as plog
from src.utils.logger import SetupCallback
from pytorch_lightning.callbacks import EarlyStopping
from src.utils.utils import process_args
import math
import wandb
from src.tools.logger import SetupCallback,BackupCodeCallback
from shutil import ignore_patterns

def create_parser():
    parser = argparse.ArgumentParser()
    
    # Set-up parameters
    parser.add_argument('--res_dir', default='/liuyunfan/qianyunhang/PFMBench/tasks/results', type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)#训练过程中每隔几轮进行一次验证，默认为1轮。
    parser.add_argument('--offline', default=1, type=int) #指定WandB是否处于离线模式，默认为1（即是）
    parser.add_argument('--seed', default=2024, type=int)
    
    parser.add_argument('--batch_size', default=32, type=int)#设置每个批次的样本数量，默认为32
    parser.add_argument('--pretrain_batch_size', default=4, type=int)#设置预训练阶段的批次大小，默认为4。
    parser.add_argument('--num_workers', default=4, type=int)#数据加载时的工作线程数量，默认为4。
    parser.add_argument('--seq_len', default=1022, type=int)#序列的长度，默认为1022
    parser.add_argument('--gpus_per_node', default=1, type=int)#每个节点的GPU数量，默认为1
    parser.add_argument('--num_nodes', default=1, type=int)#节点数量，默认为1
    
    # Training parameters
    parser.add_argument('--epoch', default=1, type=int, help='end epoch')#训练的最大轮数，默认为50
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')#学习率，ProteinInvBench: 1e-3
    parser.add_argument('--lr_scheduler', default='cosine')#onecycle
    
    # Model parameters
    parser.add_argument('--sequence_only', default=0, type=int)#添加参数 --sequence_only，默认为0，表示是否只使用序列信息（0：否，1：是）
    parser.add_argument('--finetune_type', default='adapter', type=str, choices=['adapter', 'peft'])#--finetune_type，默认值为 adapter，设置微调类型，可以选择 adapter 或 peft
    parser.add_argument('--peft_type', default='adalora', type=str, choices=['lora', 'adalora', 'ia3', 'dora', 'freeze'])#添加参数 --peft_type，默认值为 adalora，指定参数高效微调的类型
    parser.add_argument('--pretrain_model_name', default='esm2_650m', type=str, choices=[
        'esm2_650m', 'esm3_1.4b', 'esmc_600m', 'procyon', 'prollama', 'progen2', 'prostt5', 
        'protgpt2', 'protrek', 'saport', 'gearnet', 'prost', 'prosst2048', 'venusplm', 
        'prott5', 'dplm', 'ontoprotein', 'ankh_base', 'pglm', 'esm2_35m', 'esm2_150m', 
        'esm2_3b',  'esm2_15b', 'protrek_35m', 'saport_35m', 'saport_1.3b', 'dplm_150m', 'dplm_3b', 'pglm-3b',
        'StructGNN', 'GraphTrans', 'GVP', 'GCA', 'AlphaDesign', 'ESMIF', 'PiFold', 'ProteinMPNN', 'KWDesign', 'E3PiFold', 'UniIF'
    ])#--pretrain_model_name，指定使用的预训练模型，默认值为 esm2_650m，并列出了可选的模型名称。
    parser.add_argument("--config_name", type=str, default='fitness_prediction', help="Name of the Hydra config to use")#添加参数 --config_name，指定Hydra配置文件的名称，默认为 fitness_prediction。
    parser.add_argument("--metric", type=str, default='val_loss', help="metric for early stop")#指定用于早期停止监控的指标，默认为验证损失 val_loss
    parser.add_argument("--direction", type=str, default='min', help="metric direction")#指定监控指标的优化方向，默认为 min（最小化
    parser.add_argument("--enable_es", type=int, default=1, help="enable early stopping")#用于启用早期停止，默认为1（启用）
    parser.add_argument("--feature_extraction", type=int, default=0, help="perform feature extraction(paper used only)")#--feature_extraction，用于指示是否执行特征提取，默认为0（不提取）。
    parser.add_argument("--feature_save_dir", type=str, default=None, help="feature saved dir(paper used only)")#添加参数 --feature_save_dir，指定特征保存的目录，默认为 None。
   
    #############  structure-to-sequence-eval
    parser.add_argument('--pad', default=1024, type=int)
    parser.add_argument('--min_length', default=40, type=int)
    parser.add_argument('--augment_eps', default=0.0, type=float, help='noise level')
    parser.add_argument('--use_dist', default=1, type=int)
    parser.add_argument('--use_product', default=0, type=int)
    parser.add_argument('--dataset', default='CASP15') # CASP15  CATH4.2 CATH4.3
    parser.add_argument('--data_root', default='/liuyunfan/qianyunhang/PFMBench/dataset')
    ############# 
    parser.add_argument('--eval_model', default='others', type=str, choices=['structure-to-sequence-eval',  'others'])
    args = process_args(parser, config_path='../../tasks/configs')
    print(args)
    return args


def load_callbacks(args):
    callbacks = []
    
    logdir = str(os.path.join(args.res_dir, args.ex_name))#创建日志目录的路径，结合结果目录 args.res_dir 和实验名称 args.ex_name。
    
    ckptdir = os.path.join(logdir, "checkpoints")#在日志目录下创建存储模型检查点的子目录路径，命名为 checkpoints。
   
    if args.eval_model == "structure-to-sequence-eval":
        callbacks.append(BackupCodeCallback(os.path.dirname(args.res_dir),logdir, ignore_patterns=ignore_patterns('results*', 'pdb*', 'metadata*', 'vq_dataset*')))
        metric = "recovery" # Recovery Confidence Diversity sc-TM Robustness Efficiency
        sv_filename = 'best-{epoch:02d}-{recovery:.3f}'
        callbacks.append(plc.ModelCheckpoint(
            monitor=metric,
            filename=sv_filename,
            save_top_k=15,
            mode='max',
            save_last=True,
            dirpath = ckptdir,
            verbose = True,
            every_n_epochs = args.check_val_every_n_epoch,
        ))
        now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
        cfgdir = os.path.join(logdir, "configs")
        callbacks.append(
            SetupCallback(
                    now = now,
                    logdir = logdir,
                    ckptdir = ckptdir,
                    cfgdir = cfgdir,
                    config = args.__dict__,
                    argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],)
        )
        if args.lr_scheduler:
            callbacks.append(plc.LearningRateMonitor(
                logging_interval=None))
    elif args.eval_model == "others":
        metric = "val_loss" if args.metric is None else args.metric#ProteinInvBench: Recovery Confidence Diversity sc-TM Robustness Efficiency
        direction = "min" if args.direction is None else args.direction#确定监控指标的优化方向，默认为 min（最小化），可以根据 args.direction 更改。
        # metric = "val_loss"
        print(f"metric: {metric}, direction: {direction}")
        sv_filename = 'best-{epoch:02d}-{val_loss:.4f}'
        callbacks.append(plc.ModelCheckpoint(
            monitor=metric,
            filename=sv_filename,
            save_top_k=3,
            mode=direction,
            save_last=True,
            dirpath = ckptdir,
            verbose = True,
            # every_n_train_steps=args.check_val_every_n_step
            every_n_epochs = args.check_val_every_n_epoch,
        ))

        
        now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
        cfgdir = os.path.join(logdir, "configs")
        callbacks.append(
            SetupCallback(
                    now = now,
                    logdir = logdir,
                    ckptdir = ckptdir,
                    cfgdir = cfgdir,
                    config = args.__dict__,
                    argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],)
        )
        
        if args.enable_es:
            early_stop_callback = EarlyStopping(
                monitor=metric,   # 必须和你的 validation step log 出来的 key 一致
                patience=5,
                mode=direction,           # loss 下降才是“好”，因此用 min
                strict=True,
            )
            callbacks.append(early_stop_callback)
        
        if args.lr_scheduler:
            callbacks.append(plc.LearningRateMonitor(
                logging_interval=None))
    return callbacks


def automl_setup(args, logger):
    args.res_dir = os.path.join(args.res_dir, args.ex_name)
    print(wandb.run)
    args.ex_name = wandb.run.id
    wandb.run.name = wandb.run.id
    logger._save_dir = str(args.res_dir)
    os.makedirs(logger._save_dir, exist_ok=True)
    logger._name = wandb.run.name
    logger._id = wandb.run.id
    return args, logger
    

def main():
    args = create_parser()
    output_dir = '/liuyunfan/qianyunhang/PFMBench/tasks/results'
    os.makedirs(output_dir, exist_ok=True)
    
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(project='protein_benchmark', entity='3180655398-', dir=os.path.join(args.res_dir, args.ex_name))
    logger = plog.WandbLogger(
                    project = 'protein_benchmark',
                    name=args.ex_name,
                    save_dir=os.path.join(args.res_dir, args.ex_name),
                    dir = os.path.join(args.res_dir, args.ex_name),
                    offline = args.offline,
                    entity = "3180655398-")

    #================ for wandb sweep ==================
    args, logger = automl_setup(args, logger)
    #====================================================
    
    # generated a random seed
    args.seed = random.randint(1, 9999)
    print(f"Generated random seed: {args.seed}")
    pl.seed_everything(args.seed)
    gpu_count = torch.cuda.device_count()
    if args.eval_model == "others":
        data_module = DInterface(**vars(args))
    if args.eval_model == "structure-to-sequence-eval":
        data_module = DInterface_st2s(**vars(args))
  
    # here we perform feature extraction
    if args.feature_extraction and args.finetune_type == "adapter" and args.eval_model == "others":
        data_module.data_setup(target="test")
        feature_save_dir = "./feature_extraction" if not args.feature_save_dir else args.feature_save_dir
        os.makedirs(feature_save_dir, exist_ok=True)
        config_res_dir = os.path.join(feature_save_dir, args.config_name)
        os.makedirs(config_res_dir, exist_ok=True)
        result_parquet = f"{args.pretrain_model_name}-{args.config_name}.parquet"
        result_parquet = os.path.join(config_res_dir, result_parquet)
        test_dataset = data_module.test_set
        results = []
        for sdata in test_dataset.data:
            embedding = sdata['embedding'].mean(0).flatten().float().numpy()
            label = sdata['label'].numpy().item() # only for multi-label classification tasks
            results.append(
                {
                    "embedding": embedding,
                    "label": label,
                    "model": f"{args.pretrain_model_name}"
                }
            )
        results = pd.DataFrame(results)
        results.to_parquet(result_parquet, index=False, engine="pyarrow")
        print(f"[Feature extraction] Result parquet: {result_parquet}")
        return 
    
    if args.eval_model == "others":
        data_module.data_setup()
        steps_per_epoch = math.ceil(len(data_module.train_set)/args.batch_size/gpu_count)
        args.lr_decay_steps =  steps_per_epoch*args.epoch
    if args.eval_model == "structure-to-sequence-eval":
        data_module.setup()
        args.steps_per_epoch = math.ceil(len(data_module.trainset)/args.batch_size/gpu_count)
        print(f"steps_per_epoch {args.steps_per_epoch},  gpu_count {gpu_count}, batch_size{args.batch_size}")

    # data_module.data_setup()
    #gpu_count = torch.cuda.device_count()

    #steps_per_epoch = math.ceil(len(data_module.train_set)/args.batch_size/gpu_count)

    # args.steps_per_epoch = math.ceil(len(data_module.trainset)/args.batch_size/gpu_count)
    # print(f"steps_per_epoch {args.steps_per_epoch},  gpu_count {gpu_count}, batch_size{args.batch_size}")

    #args.lr_decay_steps =  steps_per_epoch*args.epoch
    if args.eval_model == "others":      
        model = MInterface(**vars(args))
        data_module.MInterface = model
    if args.eval_model == "structure-to-sequence-eval":
        model = MInterface_st2s(**vars(args))

    if args.eval_model == "structure-to-sequence-eval":
        ckpt = torch.load('/liuyunfan/qianyunhang/PFMBench/model_zoom/PDB/AlphaDesign/checkpoint.pth')
        ckpt = {k.replace('_forward_module.model.',''):v for k,v in ckpt.items()}
        model.model.load_state_dict(ckpt)

   # data_module.MInterface = model
    # callbacks = load_callbacks(args)
    # trainer_config = {
    #     "accelerator": "gpu",
    #     'devices': gpu_count,  # Use gpu count
    #     'max_epochs': args.epoch,  # Maximum number of epochs to train for
    #     'num_nodes': args.num_nodes,  # Number of nodes to use for distributed training
    #     "strategy": 'deepspeed_stage_2', # 'ddp', 'deepspeed_stage_2
    #     "precision": 'bf16', # structure-to-sequence-eval：32
    #     'accelerator': 'gpu',  # Use distributed data parallel
    #     'callbacks': callbacks,
    #     'logger': logger,
    #     'gradient_clip_val':1.0,
    # }

    # trainer_opt = argparse.Namespace(**trainer_config)
    
    # trainer = Trainer(**vars(trainer_opt))
    
    if args.eval_model == "structure-to-sequence-eval":
        callbacks = load_callbacks(args)
        trainer_config = {
        "accelerator": "gpu",
        'devices': gpu_count,  # Use gpu count
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': args.num_nodes,  # Number of nodes to use for distributed training
        "strategy": 'deepspeed_stage_2', # 'ddp', 'deepspeed_stage_2
        "precision": '32', # structure-to-sequence-eval：32
        'accelerator': 'gpu',  # Use distributed data parallel
        'callbacks': callbacks,
        'logger': logger,
        'gradient_clip_val':1.0,}

        trainer_opt = argparse.Namespace(**trainer_config)
        
        trainer = Trainer(**vars(trainer_opt))
        print(trainer_config)
        trainer.test(datamodule=data_module, model=model)
    if args.eval_model == "others":
        callbacks = load_callbacks(args)
        trainer_config = {
            "accelerator": "gpu",
            'devices': gpu_count,  # Use gpu count
            'max_epochs': args.epoch,  # Maximum number of epochs to train for
            'num_nodes': args.num_nodes,  # Number of nodes to use for distributed training
            "strategy": 'deepspeed_stage_2', # 'ddp', 'deepspeed_stage_2
            "precision": 'bf16', # structure-to-sequence-eval：32
            'accelerator': 'gpu',  # Use distributed data parallel
            'callbacks': callbacks,
            'logger': logger,
            'gradient_clip_val':1.0,
        }
        trainer_opt = argparse.Namespace(**trainer_config)
        trainer = Trainer(**vars(trainer_opt))
        trainer.fit(model, data_module)
        checkpoint_callback = callbacks[0]
        print(f"Best model path: {checkpoint_callback.best_model_path}")

        # 载入最佳模型
        model_state_path = os.path.join(checkpoint_callback.best_model_path, "checkpoint", "mp_rank_00_model_states.pt")
        state = torch.load(model_state_path, map_location="cuda:0")
        model.load_state_dict(state['module'])

        # 进行测试
        results = trainer.test(model, datamodule=data_module)
        # 打印测试结果
        print(f"Test Results: {results}")


if __name__ == "__main__":
    main()
    

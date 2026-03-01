from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
import warnings
import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
#from pandas.core.frame import DataFrame
from tqdm import tqdm

warnings.filterwarnings("ignore")
t_float = torch.float64
torch.multiprocessing.set_sharing_strategy('file_system')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR = DATA_DIR / "model_saved"
PREDICTION_DIR = DATA_DIR / "prediction"

for directory in [MODEL_DIR, PREDICTION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class Args:
    def __init__(self, gpu=0, subtask="regression"):
        # device
        self.gpu = str(gpu)
        self.device = 'cpu'
        # data settings
        adj_threshold = 0.1
        self.adj_str = str(int(100*adj_threshold))
        self.pos_adj_dir = "pos_adj_" + self.adj_str
        self.neg_adj_dir = "neg_adj_" + self.adj_str
        self.feat_dir = "features"
        self.label_dir = "label"
        self.mask_dir = "mask"
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        self.pre_data = pre_data
        # epoch settings
        self.max_epochs = 60
        self.epochs_eval = 10
        # learning rate settings
        self.lr = 0.0002
        self.gamma = 0.3
        # model settings
        self.hidden_dim = 128
        self.num_heads = 8
        self.out_features = 32
        self.model_name = "StockHeteGAT"
        self.batch_size = 1
        self.loss_fcn = mse_loss
        self.target_dim = 1
        self.predictor_activation = None
        # save model settings
        self.save_path = str(MODEL_DIR)
        self.load_path = self.save_path
        self.save_name = self.model_name + "_hidden_" + str(self.hidden_dim) + "_head_" + str(self.num_heads) + \
                         "_outfeat_" + str(self.out_features) + "_batchsize_" + str(self.batch_size) + "_adjth_" + \
                         str(self.adj_str)


        self.epochs_save_by = 60
        self.sub_task = subtask
        eval("self.{}".format(self.sub_task))()

    def regression(self):
        self.save_name = self.save_name + "_reg_rank_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_regression"
        self.mask_dir = self.mask_dir + "_regression"
        self.predictor_activation = None

    def regression_binary(self):
        self.save_name = self.save_name + "_reg_binary_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"
        self.predictor_activation = None

    def classification_binary(self):
        self.save_name = self.save_name + "_clas_binary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"
        self.predictor_activation = 'sigmoid'

    def classification_tertiary(self):
        self.save_name = self.save_name + "_clas_tertiary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_threeclass"
        self.mask_dir = self.mask_dir + "_threeclass"
        self.predictor_activation = 'sigmoid'




def fun_train_predict(
    data_start,
    data_middle,
    data_end,
    pre_data=None,
    prediction_horizon=None,
    target_dim=None,
    prediction_filename="pred.csv",
):
    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(f"No training data found in {TRAIN_DATA_DIR}. Please run the data generation utilities first.")

    if pre_data is None:
        if data_middle <= 0 or data_middle > len(data_files):
            raise ValueError("data_middle index is out of range for available training samples.")
        pre_data_value = Path(data_files[data_middle - 1]).stem
    else:
        pre_data_value = pre_data

    globals()['pre_data'] = pre_data_value
    args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    dataset = AllGraphDataSampler(base_dir=str(TRAIN_DATA_DIR), data_start=data_start,
                                  data_middle=data_middle, data_end=data_end)
    val_dataset = AllGraphDataSampler(base_dir=str(TRAIN_DATA_DIR), mode="val", data_start=data_start,
                                      data_middle=data_middle, data_end=data_end)
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty. Check data_start and data_middle indices.")
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x)
    val_dataset_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True)

    _, _, _, sample_labels, _ = extract_data(dataset[0], args.device)
    if sample_labels.dim() <= 1:
        inferred_dim = 1
    else:
        inferred_dim = sample_labels.shape[-1]

    if prediction_horizon is not None:
        if target_dim is not None and int(target_dim) != int(prediction_horizon):
            raise ValueError("prediction_horizon and target_dim refer to the same dimension and must match if both are provided.")
        target_dim = int(prediction_horizon)

    if target_dim is None:
        target_dim = inferred_dim
    else:
        target_dim = int(target_dim)
        if target_dim != inferred_dim:
            raise ValueError(
                f"Configured target dimension ({target_dim}) does not match the dataset label dimension ({inferred_dim})."
            )

    if target_dim < 1:
        raise ValueError("target_dim must be a positive integer.")

    args.target_dim = target_dim
    args.save_name = args.save_name + f"_tdim_{args.target_dim}"
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                                  out_features=args.out_features,
                                  predictor_out_dim=args.target_dim,
                                  predictor_activation=args.predictor_activation).to(args.device)

    # train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cold_scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    default_scheduler = cold_scheduler
    print('start training')
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(epoch=epoch, args=args, model=model, dataset_train=dataset_loader,
                                 optimizer=optimizer, scheduler=default_scheduler, loss_fcn=args.loss_fcn)
        if epoch % args.epochs_eval == 0:
            eval_loss, _ = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=args.loss_fcn)
            print('Epoch: {}/{}, train loss: {:.6f}, val loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss,
                                                                              eval_loss))
        else:
            print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss))
        if (epoch + 1) % args.epochs_save_by == 0:
            print("save model!")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, Path(args.save_path) / f"{pre_data_value}_epoch_{epoch + 1}.dat")

    # predict
    checkpoint = torch.load(Path(args.load_path) / f"{pre_data_value}_epoch_{epoch + 1}.dat")
    model.load_state_dict(checkpoint['model'])
    data_code = sorted([p.name for p in DAILY_STOCK_DIR.glob("*.csv")])
    data_code_last = data_code[data_middle:data_end]
    df_score = pd.DataFrame()
    for i in tqdm(range(len(val_dataset))):
        df = pd.read_csv(DAILY_STOCK_DIR / data_code_last[i], dtype=object)
        tmp_data = val_dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)
        model.eval()
        logits = model(features, pos_adj, neg_adj)
        predictions = logits.detach().cpu().numpy()
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        if predictions.shape[1] == 1:
            score_columns = ["score"]
        else:
            score_columns = [f"score_t+{idx + 1}" for idx in range(predictions.shape[1])]
        scores_df = pd.DataFrame(predictions, columns=score_columns)
        df = df.reset_index(drop=True)
        combined_df = pd.concat([df, scores_df], axis=1)
        df_score = pd.concat([df_score, combined_df], ignore_index=True)

        #df.to_csv('prediction/' + data_code_last[i], encoding='utf-8-sig', index=False)
    prediction_path = PREDICTION_DIR / prediction_filename
    df_score.to_csv(prediction_path, index=False)
    print(df_score)
    print(f"Saved predictions to {prediction_path.resolve()}")

    return df_score


if __name__ == "__main__":
    data_start = 20
    data_middle = 39
    data_end = data_middle + 4
    fun_train_predict(data_start, data_middle, data_end)
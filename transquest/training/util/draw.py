
import pandas as pd
from sklearn.metrics import mean_absolute_error

# HANNA'S CHANGES
#from examples.sentence_level.wmt_2020_task2.common.util import fit
#from transquest.algo.monotransquest.evaluation import pearson_corr, spearman_corr, rmse
import sys
import matplotlib.pyplot as plt

# COLAB
#PROJECT_PATH='drive/MyDrive/Adversarial_MTQE/'

# LAB MACHINE
#PROJECT_PATH='/vol/bitbucket/hsb20/'

#sys.path.append(PROJECT_PATH+'CODE/transquest/algo/sentence_level/monotransquest')
from CODE.transquest.algo.sentence_level.monotransquest.evaluation import pearson_corr, spearman_corr, rmse

#sys.path.remove(PROJECT_PATH+'CODE/transquest/algo/sentence_level/monotransquest')
#sys.path.append(PROJECT_PATH+'CODE/training/HTER/util')
from CODE.transquest.training.util.normalizer import fit


from PIL import Image

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


def draw_scatterplot(data_frame, real_column, prediction_column, path, topic):
    data_frame = data_frame.sort_values(real_column)
    sort_id = list(range(0, len(data_frame.index)))
    data_frame['id'] = pd.Series(sort_id).values

    data_frame = fit(data_frame, real_column)
    data_frame = fit(data_frame, prediction_column)

    pearson = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    mae = mean_absolute_error(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.4f\nMAE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (rmse_value, mae, pearson, spearman)

    plt.figure()
    ax = data_frame.plot(kind='scatter', x='id', y=real_column, color='DarkBlue', label='z_mean', title=topic)
    ax = data_frame.plot(kind='scatter', x='id', y=prediction_column, color='DarkGreen', label='predicted z_mean',
                    ax=ax)
    ax.text(0.5*data_frame.shape[0], min(min(data_frame[real_column].tolist()), min(data_frame[prediction_column].tolist())), textstr, fontsize=10)

    fig = ax.get_figure()
    fig.savefig(path)

    return

def draw_scatterplot_multitransquest(data_frame, real_column, prediction_column, path, topic, curr_task):
    data_frame = data_frame.sort_values(real_column)
    sort_id = list(range(0, len(data_frame.index)))
    data_frame['id'] = pd.Series(sort_id).values

    data_frame = fit(data_frame, real_column)
    data_frame = fit(data_frame, prediction_column)

    pearson = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    mae = mean_absolute_error(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.4f\nMAE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (rmse_value, mae, pearson, spearman)

    plt.figure()
    ax = data_frame.plot(kind='scatter', x='id', y=real_column, color='DarkBlue', label='z_mean', title=topic)
    ax = data_frame.plot(kind='scatter', x='id', y=prediction_column, color='DarkGreen', label='predicted z_mean',
                    ax=ax)
    ax.text(0.5*data_frame.shape[0], min(min(data_frame[real_column].tolist()), min(data_frame[prediction_column].tolist())), textstr, fontsize=10)

    fig = ax.get_figure()
    fig.savefig(path)

    img = Image.open(path)
    #wandb.log({"pred_scatterplot" + str(curr_task): wandb.Image(img)})

    return

def print_stat(data_frame, real_column, prediction_column):
    data_frame = data_frame.sort_values(real_column)

    pearson = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    mae = mean_absolute_error(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.4f\nMAE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (rmse_value, mae, pearson, spearman)

    print(textstr)
import numpy as np
from sklearn.metrics import r2_score


class Args:
    """
        Prediction Model Parameters
    """
    def __init__(self):
        self.train_length = 7
        self.valid_length = 2
        self.test_length = 1

        self.train_ratio = self.train_length / (self.train_length + self.valid_length + self.test_length)
        self.valid_ratio = self.valid_length / (self.train_length + self.valid_length + self.test_length)
        self.test_ratio = 1 - self.train_ratio - self.valid_ratio
        
        self.train                  = True    
        self.evaluate               = True
        self.window_size            = 24     # time step
        self.horizon                = 12     # prediction horizon

        self.epoch                  = 50
        self.lr                     = 1e-4
        self.multi_layer            = 5
        self.device                 = 'cpu'
        self.validate_freq          = 1
        self.batch_size             = 32
        self.norm_method            = 'z_score'
        self.optimizer              = 'RMSProp'
        self.early_stop             = False
        self.exponential_decay_step = 5

        # attention
        self.decay_rate             = 0.5
        self.dropout_rate           = 0.5
        self.leakyrelu_rate         = 0.2

        self.stack_cnt              = 2     # TGCN和STEMGNN需要


def masked_MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mask = (v == 0)
    percentage = np.abs(v_ - v) / np.abs(v)
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)


def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mape = (np.abs(v_ - v) / np.abs(v)+1e-5).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)

def R_squared(v, v_, axis=None):
    '''
    R Squared
    '''
    rss = np.sum((v - v_) ** 2, axis=axis).astype(np.float64)
    tss = np.sum((v - np.mean(v)) ** 2, axis=axis).astype(np.float64)
    return 1 - rss / tss


def evaluate(y, y_hat, by_step=False, by_node=False):
    '''
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    '''
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat), R_squared(y, y_hat)
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0), R_squared(y, y_hat, axis=0)
    if by_step:
        return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2)), R_squared(y, y_hat, axis=(0, 2))
    if by_node:
        return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1)), R_squared(y, y_hat, axis=(0, 1))
    


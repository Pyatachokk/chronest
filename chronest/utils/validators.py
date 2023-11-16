import polars as pl
from chronest.utils.constants import DATE_COLUMN

def validate_y(func):
    def wrapper_func(y: pl.DataFrame, *args, **kwargs):

        if y.find_idx_by_name('x'):
            print('kek')

        func(y=y, *args, **kwargs)

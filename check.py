import h5py
with h5py.File("/home/users/zhangqn8/storage/20min_processed_signals_fullimage.h5", "r") as f:
    print(list(f.keys()))

# CUDA_VISIBLE_DEVICES=3 python infer_mixed_dataset.py --dataset 'HAMMER' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --method 'priorda_zs_518x518' --encoder 'vitl' --split '/home/robotarm/object_depth_percetion/dataset/splits/HAMMER_test.txt'

# CUDA_VISIBLE_DEVICES=3 python infer_mixed_dataset.py --dataset 'HouseCat6D' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --method 'priorda_zs_518x518' --split '/home/robotarm/object_depth_percetion/dataset/splits/housecat6d_test.txt'

# CUDA_VISIBLE_DEVICES=3 python infer_mixed_dataset.py --dataset 'PhoCAL' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --method 'priorda_zs_518x518' --split '/home/robotarm/object_depth_percetion/dataset/splits/PhoCAL_test.txt'

# CUDA_VISIBLE_DEVICES=3 python infer_mixed_dataset.py --dataset 'TransCG' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --method 'priorda_zs_518x518' --camera 'd435' --split '/home/robotarm/object_depth_percetion/dataset/splits/TransCG_d435_test.txt'

# CUDA_VISIBLE_DEVICES=3 python infer_mixed_dataset.py --dataset 'XYZ-IBD' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --method 'priorda_zs_518x518' --camera 'd435' --split '/home/robotarm/object_depth_percetion/dataset/splits/XYZ-IBD_test.txt'

# CUDA_VISIBLE_DEVICES=3 python infer_mixed_dataset.py --dataset 'YCB-V' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --method 'priorda_zs_518x518' --camera 'd435' --split '/home/robotarm/object_depth_percetion/dataset/splits/YCB-V_test.txt'

# CUDA_VISIBLE_DEVICES=3 python infer_mixed_dataset.py --dataset 'T-LESS' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --method 'priorda_zs_518x518' --camera 'd435' --split '/home/robotarm/object_depth_percetion/dataset/splits/T-LESS_test_primesense.txt'

CUDA_VISIBLE_DEVICES=4 python infer_mixed_dataset.py --dataset 'GN-Trans' --dataset_root '/data/robotarm/dataset' --img_height 518 --img_width 518 --method 'priorda_zs_518x518' --camera 'd435' --split '/home/robotarm/object_depth_percetion/dataset/splits/GN-Trans_test.txt'

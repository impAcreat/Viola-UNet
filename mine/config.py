## from load_model
wind_levels = [[15,85], [-15, 200],[-100, 1300]]
spacing = [1., 1., 3.] 

## paths: relative to folder "mine"
data_input_folder = "../../INSTANCE/train_2/data/"
label_input_folder = "../../INSTANCE/train_2/label/"
dn_model_path = "./experiment/dn_model.pth"

## data
train_ratio = 0.8

## params
batch_size = 1
num_workers = 4     # 控制用于数据加载的子进程数量: more workers --> faster
num_epochs = 25


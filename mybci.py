import argparse, os
from train import train, training
from preprocessing import preprocess

def get_data_path():
    data_path = os.path.dirname(os.path.abspath(__file__)) + "/data_path.txt"
    if not os.path.exists(data_path):
        raise Exception("data_path don t exist!")
    with open(data_path, 'r') as file:
        lines = file.readlines()

    folder_path = ''.join(line.strip() for line in lines)
    if not os.path.exists(folder_path):
        raise Exception("folder_path don t exist!")
    return folder_path

def args_parse():
    parser = argparse.ArgumentParser(description="Train a neural network with specified parameters.")

    parser.add_argument("--subject_num", type=int, default=0, help="Subject number to process (default: 0 for all subjects)")
    parser.add_argument("--task_num", type=int, default=0, help="Task number to process (default: 0 for all tasks)")
    parser.add_argument("--plot", action='store_true', help="Plot variable (call for plotting)")
    parser.add_argument("--train", action='store_true', help="Train call")
    parser.add_argument("--predict", action='store_true', help="Predict call")

    return parser.parse_args()

def check_args(args):
    if args.subject_num > 109 or args.subject_num < 0:
        raise Exception(f"We got 109 subjects and u entered {args.subject_num}")
    if args.task_num > 14 or args.task_num < 0:
        raise Exception(f"We got 14 subjects and u entered {args.task_num}")
    if args.predict or args.plot:
        if args.subject_num == 0 or args.task_num == 0:
            raise Exception("You need to select the subject and the task for predcition or plotting!")

def lanch_tasks(preprocessmodule : preprocess, train_module : train, args, folder_path):
    if not args.predict:
        if args.subject_num > 0:
            subject_path = f"{folder_path}/S{args.subject_num:03}"
            if args.task_num > 0:
                raise Exception("It s stupid to train the module on one task lol")
            _, scores = training(preprocessmodule, train_module, subject_path)
            print(scores)
            print(f'cross_val_score: {scores.mean()}')
        else:
            accuracy, _ = training(preprocessmodule, train_module, folder_path, True)
            print(f'Test Data accuracy: {accuracy:.4f}')

def lanch_model(args):
    preprocessmodule = preprocess(args.plot)
    folder_path = get_data_path()

    if not args.plot:
        train_module = train()
        kwargs = {
            "preprocessmodule":preprocessmodule,
            "train_module":train_module,
            "folder_path":folder_path,
            "args":args
        }
        lanch_tasks(**kwargs)
    else:
        print("Plotting process...")
        if args.task_num == 1 or args.task_num == 2:
            preprocessmodule.set_dict(dict(T0=0))
        file_path = f"{folder_path}/S{args.subject_num:03}/S{args.subject_num:03}R{args.task_num:02}.edf"
        if not os.path.exists(file_path):
            raise Exception("file_path don t exist!")
        preprocessmodule.process(file_path)

if __name__ == '__main__':
    args = args_parse()
    lanch_model(args)
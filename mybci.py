import argparse, os
from train import training
from preprocessing import preprocess
from predict import predict

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
    parser = argparse.ArgumentParser(description="Train an EEG model with specified parameters.")

    parser.add_argument("--plot", type=int, nargs='+', default=[], help="Plot Task (first arg is subject_num and second is task_num)")
    parser.add_argument("--train", type=int, nargs='+', default=[], help="Train call for multiple subjects")
    parser.add_argument("--predict", type=int, nargs='+', default=[], help="Predict Task (first arg is subject_num and second is task_num)")
    parser.add_argument("--multiple_classifiers", action='store_true', help="Call in train for multi classifiers test!")
    return parser.parse_args()

def check_args(args):
    if sum(bool(x) for x in [args.plot, args.train, args.predict]) > 1:
        raise Exception(f"You should call plot or train or predict!")
    if args.multiple_classifiers and len(args.train) == 0:
        raise Exception("multiple_classifiers arg is called just with --train")
    for arg in [args.plot, args.predict]:
        if len(arg) > 0:
            if len(arg) > 2:
                raise Exception(f"Argument called has to have just two args!")
            subject_num = arg[0]
            task_num = arg[1]
            if subject_num > 109 or subject_num < 0:
                raise Exception(f"We got 109 subjects and u entered {subject_num}")
            if task_num > 14 or task_num < 0:
                raise Exception(f"We got 14 tasks and u entered {task_num}")
    if len(args.train) > 0:
        for subject_num in args.train:
            if subject_num > 109 or subject_num < 0:
                raise Exception(f"We got 109 subjects and u entered {subject_num}")


def lanch_tasks(preprocessmodule : preprocess, args, folder_path):
    if len(args.predict) == 0:
        if len(args.train) > 0:
            subjects_path = [f"{folder_path}/S{subject_num:03}" for subject_num in args.train]
            scores = training(preprocessmodule, subjects_path, args.multiple_classifiers)
            if scores:
                print(scores)
                print(f'cross_val_score: {scores.mean()}')
        else:
            training(preprocessmodule, folder_path, True)
    else:
        predict(preprocessmodule, args.predict, folder_path)

def lanch_model(args):
    preprocessmodule = preprocess()
    folder_path = get_data_path()

    if not args.plot:
        kwargs = {
            "preprocessmodule":preprocessmodule,
            "folder_path":folder_path,
            "args":args
        }
        lanch_tasks(**kwargs)
    else:
        print("Plotting process...")
        preprocessmodule.set_plot(True)
        file_path = f"{folder_path}/S{args.plot[0]:03}/S{args.plot[0]:03}R{args.plot[1]:02}.edf"
        if args.plot[1] in [1, 2]:
            preprocessmodule.set_dict(dict(T0=0))
        if not os.path.exists(file_path):
            raise Exception("file_path don t exist!")
        preprocessmodule.process(file_path)

if __name__ == '__main__':
    args = args_parse()
    check_args(args)
    lanch_model(args)

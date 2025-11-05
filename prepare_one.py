import os
from MetalBindProcessor import MetalBindProcessor
from default_config.dir_options import dir_opts
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Network parameters")

    # Main parameters
    parser.add_argument(
        "--ligand", type=str, help="specific one ligand",
        choices=['FE', 'ZN', 'CA', 'MN', 'MG', 'CD', 'CO', 'CU', 'CU1', 'FE2', 'K', 'NA', 'NI'], required=True,
    )
    parser.add_argument(
        "--pdbid",
        type=str,
        default="",
        help="Specify a PDB to process. Or the program will process the PDBs in defaut list one by one",
    )
    parser.add_argument(
        "--RQ_window",
        type=float,
        default="6.0",
        help="RQ window Radius to use for the convolution",
    )
    return parser
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.ligand == 'CA':
        train_file = 'Dataset_lists/CA-1670_Train.txt'
        test_file = 'Dataset_lists/CA-415_Test.txt'
    if args.ligand == 'MN':
        train_file = 'Dataset_lists/MN-578_Train.txt'
        test_file = 'Dataset_lists/MN-144_Test.txt'
    if args.ligand == 'MG':
        train_file = 'Dataset_lists/MG-1897_Train.txt'
        test_file = 'Dataset_lists/MG-465_Test.txt'
    if args.ligand == 'FE':
        train_file = 'Dataset_lists/FE-278_Train.txt'
        test_file = 'Dataset_lists/FE-70_Test.txt'
    if args.ligand == 'ZN':
        train_file = 'Dataset_lists/ZN-1966_Train.txt'
        test_file = 'Dataset_lists/ZN-484_Test.txt'
    if args.ligand == 'CD':
        train_file = 'Dataset_lists/CD-36_Train.txt'
        test_file = 'Dataset_lists/CD-9_Test.txt'
    if args.ligand == 'CO':
        train_file = 'Dataset_lists/CO-226_Train.txt'
        test_file = 'Dataset_lists/CO-55_Test.txt'
    if args.ligand == 'CU':
        train_file = 'Dataset_lists/CU-157_Train.txt'
        test_file = 'Dataset_lists/CU-39_Test.txt'
    if args.ligand == 'CU1':
        train_file = 'Dataset_lists/CU1-46_Train.txt'
        test_file = 'Dataset_lists/CU1-11_Test.txt'
    if args.ligand == 'FE2':
        train_file = 'Dataset_lists/FE2-119_Train.txt'
        test_file = 'Dataset_lists/FE2-30_Test.txt'
    if args.ligand == 'K':
        train_file = 'Dataset_lists/K-50_Train.txt'
        test_file = 'Dataset_lists/K-13_Test.txt'
    if args.ligand == 'NA':
        train_file = 'Dataset_lists/NA-79_Train.txt'
        test_file = 'Dataset_lists/NA-20_Test.txt'
    if args.ligand == 'NI':
        train_file = 'Dataset_lists/NI-42_Train.txt'
        test_file = 'Dataset_lists/NI-11_Test.txt'

    if not os.path.exists('Dataset/'):
        os.mkdir('Dataset/')
    base_dir = 'Dataset/'+ args.ligand+'/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dir_opts = dir_opts(base_dir)

    protein_list = []
    with open(train_file, 'r') as pid:
        for line in pid.readlines():
            protein_list.append(line.strip())

    with open(test_file, 'r') as pid:
        for line in pid.readlines():
            protein_list.append(line.strip())
    print(len(protein_list))
    if not os.path.exists(dir_opts['data_label']):
        os.mkdir(dir_opts['data_label'])
    for index, item in enumerate(protein_list):
        pair = item.split('\t')[0]
        anno = item.split('\t')[1] if len(item.split('\t'))>1 else None
        if args.pdbid != '' and args.pdbid!= pair:
            continue
        #if not specify the binding sites given in residue id. GeoBind will automatically compute the Binding sites
        if os.path.exists(os.path.join(dir_opts['data_label'], pair)):
            continue
        print(index, item, pair, anno)
        try:
            rbp=MetalBindProcessor(pair, anno, args.ligand, dir_opts, RQ_window=args.RQ_window)
            rbp.get_data()
        except Exception as e:
            print(f"[ERROR]: {pair}：{type(e).__name__}: {e}")
            raise
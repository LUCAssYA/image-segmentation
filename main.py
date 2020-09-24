import argparse
import make_pytable
import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i_path", '--img_path',  help = "input img path", nargs = '*', type = str)
    parser.add_argument('-m_path', '--mask_path', help = 'input mask path', nargs = '*', type = str)
    parser.add_argument('-p', '--patch_size', help = 'input patch size', default = 512, type = int)
    parser.add_argument('-s', '--stride_size',  help = 'input stride size', default = 256, type = int)
    parser.add_argument('-pad', '--pad_size', help = 'input padding size', default = 256, type = int)
    parser.add_argument('-split', '--split', help = 'input train test split', default = 0.1)
    parser.add_argument('-n', '--n_classes', help = 'input number of class', default = 1, type = int)
    parser.add_argument('-i_type', '--img_type',  help = 'input img type', default = 'tif', type = str)
    parser.add_argument('-m_type', '--mask_type', help = 'input mask type', default = 'png', type = str)
    parser.add_argument('-lr', '--learning_rate', help = 'input learning rate', type = int)
    parser.add_argument('-e', '--epochs', help = 'input epochs', type = int)
    parser.add_argument('-g', '--gpuid', help = 'input gpuid', default = '0', type = str)
    parser.add_argument('-m', '--model', help = 'input model name', type = str)
    parser.add_argument('-b', '--batch_size', help = 'input_batch_size', default = 8, type = int)

    args = parser.parse_args()




    make_pytable.make_pytable(img_path = args.img_path, label_path = args.mask_path, patch_size = args.mask_path,
                              stride_size = args.stride_size, pad_size = args.pad_size, split = args.split,
                              num_classes = args.n_classes, imgtype = args.img_type, labeltype = args.mask_type)

    train.train(lr = args.learning_rate, patch_size = args.patch_size, n_classes = args.n_classes,
                epochs = args.epochs, gpuid = args.gpuid, model_name = args.model, batch_size = args.batch_size)
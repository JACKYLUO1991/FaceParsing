import argparse


def str2bool(v):
    return v.lower() in ('true')


def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imsize', type=int, default=512)
    parser.add_argument(
        '--arch', type=str, choices=['UNet', 'DFANet', 'DANet', 'DABNet', 'CE2P', 'FaceParseNet18',
                                     'FaceParseNet34', "FaceParseNet50", "FaceParseNet101", "EHANet18"], required=True)

    # Training setting
    parser.add_argument('--epochs', type=int, default=200,
                        help='how many times to update the generator')
    parser.add_argument('--pretrained_model', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--g_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--classes', type=int, default=19)

    # Testing setting
    # parser.add_argument('--test_size', type=int, default=2824)
    # parser.add_argument('--val_size', type=int, default=2993)
    parser.add_argument('--model_name', type=str, default='model.pth')

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)

    # Path
    parser.add_argument('--img_path', type=str,
                        default='./Data_preprocessing/train_img')
    parser.add_argument('--label_path', type=str,
                        default='./Data_preprocessing/train_label')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--val_img_path', type=str,
                        default='./Data_preprocessing/val_img')
    parser.add_argument('--val_label_path', type=str,
                        default='./Data_preprocessing/val_label')
    parser.add_argument('--test_image_path', type=str,
                        default='./Data_preprocessing/test_img')
    parser.add_argument('--test_label_path', type=str,
                        default='./Data_preprocessing/test_label')
    parser.add_argument('--test_color_label_path', type=str,
                        default='./test_color_visualize')

    # Step size
    parser.add_argument('--sample_step', type=int, default=200)
    parser.add_argument('--tb_step', type=int, default=100)

    return parser.parse_args()

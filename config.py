
def parse_arguments(parser):
    ospath='E:/quz/early_detection_/early_detection'
    train =ospath+'/Data/weibo/train.pickle'
    test = ospath+'/Data/weibo/test.pickle'
    output = ospath+'/Data/weibo/output/'
    parser.add_argument('--training_file', type=str, default=train, help='')
    #parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('--testing_file', type=str, default=test, help='')
    parser.add_argument('--output_file', type=str, default=output, help='')
    # parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    # #parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    # parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    # parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default = 32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=20, help='')
    parser.add_argument('--lambd', type=int, default= 1, help='')
    parser.add_argument('--text_only', type=bool, default= True, help='')

    parser.add_argument('--input_size', type = int, default = 28, help = '')
    parser.add_argument('--hidden_size', type = int, default = 256, help = '')
    parser.add_argument('--num_layers', type = int, default = 2, help = '')
    parser.add_argument('--num_classes', type = int, default = 10, help = '')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=20, help='')
    parser.add_argument('--num_epochs', type=int, default=50, help='')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')
    return parser
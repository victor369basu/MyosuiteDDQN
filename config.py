import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # DDQ-Learning Hyper-parameters
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help="Set environmnet seed.")
    parser.add_argument('-me','--min_episodes', dest='min_episodes', default=3, type=int,
                        help="Minimum number of episodes in order to aggregate enough data after which training would start.")
    parser.add_argument('-ur','--update_repeats', dest='update_repeats', default=50, type=int,
                        help="the Q-Network is trained update_repeats many times with a batch of size batch_size from the memory.")
    parser.add_argument('-ne','--num_episodes', dest='num_episodes', default=300, type=int,
                        help="the number of episodes played in total.")
    parser.add_argument('-mms','--max_memory_size', dest='max_memory_size', default=50000, type=int,
                        help="size of the replay memory.")
    parser.add_argument('-ms','--measure_step', dest='measure_step', default=10, type=int,
                        help="every measure_step episode the performance is measured.")
    parser.add_argument('-mr','--measure_repeats', dest='measure_repeats', default=100, type=int,
                        help="the amount of episodes played in to asses performance.")

    # Model and Transformer hyper-parameters
    parser.add_argument('-hd','--hidden_dim', dest='hidden_dim', default=100, type=int,
                        help="hidden dimensions for the Q_network.")
    parser.add_argument('-dm','--d_model', dest='d_model', default=256, type=int,
                        help="the number of expected features in the encoder/decoder inputs.")
    parser.add_argument('-nt','--ntoken', dest='ntoken', default=256, type=int,
                        help="number of tokens")
    parser.add_argument('-nh','--nhead', dest='nhead', default=8, type=int,
                        help="the number of heads in the multiheadattention models")
    parser.add_argument('-nl','--nlayers', dest='nlayers', default=2, type=int,
                        help="the number of sub-encoder-layers in the encoder.")
    parser.add_argument('-drop','--dropout', dest='dropout', default=0.0, type=float,
                        help="the dropout value.")
    parser.add_argument('-ev','--env_name', dest='env_name', default="myoHandReachFixed-v0", type=str,
                        help="name of the gym environment.")
    parser.add_argument('-g', '--gamma', dest='gamma', default=0.99, type=float,
                        help="reward discount factor.")

    # Training parameters
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', default=0.0001, type=float,
                        help="Learning Rate.")
    parser.add_argument('-ep', '--eps', dest='eps', default=0.09, type=float,
                        help="epsilon value of optimizer.")
    parser.add_argument('-e', '--epochs', dest='epochs',type=int, default=100, help="Training Epochs.")
    parser.add_argument('-b', '--batch_size', dest='batch_size',type=int, default=64,
                        help="Batch size during each training step.")
    parser.add_argument('-l', '--loss_fn', dest='loss_fn',type=str, default='mse',choices=['mse', 'cel'],
                        help="select loss function.")
    parser.add_argument('-t','--train', type=str2bool, default=False,help="True when train the model, \
                        else used for testing.")
    # Validation
    parser.add_argument('--idx', type=int, default=0,
                        help="Validation index")
    # model save directory
    parser.add_argument('-path', '--model_save_path', dest='path', type=str, default='./models/',
                        help="Path to Saved model directory.")
    return parser.parse_args()



CONFIG_NUM_EPOCHS = 2

CONFIG_RAW_JSON_PATH = "/Users/admin/PycharmProjects/rezapycharmprojects/gradient-muscle/workoutdata/rezadata/"

#need to make these next three
CONFIG_NN_PICKLES_PATH = "/Users/admin/Desktop/tmp/nnpickles_folder/"
CONFIG_NORMALIZED_NN_PICKLES_PATH = "/Users/admin/Desktop/tmp/normalized_pickles_folder/"
CONFIG_NORMALIZE_VALS_PATH = "/Users/admin/Desktop/tmp/normvals"

CONFIG_WORKOUT_LOOKBACK = 1
CONFIG_BATCH_SIZE = 2
CONFIG_MAX_REPS_PER_SET = 15
CONFIG_MAX_WEIGHT = 1000
#assuming they do 25 sets per workout, everyday
#its 40 days before there are 1000 sets
#and after 1000 timesteps the LSTM doesnt really remember further
CONFIG_DAYS_SINCE_LAST_WORKOUT_CAP = 7*6
'''
MIT License

Copyright (c) 2017 Reza Hussain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


STARTING_REPS = 6
MINIMUM_WEIGHT = 45.0
MAXIMUM_WEIGHT = 1000.0

CONFIG_NUM_EPOCHS = 20000

CONFIG_MODEL_NAME = "stressmodel"
CONFIG_SAVE_MODEL_LOCATION = "./tmp/"

CONFIG_RAW_JSON_PATH = "./workoutdata/"



#need to make these next three

CONFIG_NN_BODY_MODEL_PICKLES_PATH = "./tmp/nnpickles_body/"
CONFIG_NN_STRESS_MODEL_PICKLES_PATH = "./tmp/nnpickles_stress/"

CONFIG_NORMALIZE_VALS_PATH = "./tmp/normvals"

CONFIG_WORKOUT_LOOKBACK = 2
CONFIG_BATCH_SIZE = 9
CONFIG_MIN_REPS_PER_SET = 3
CONFIG_MAX_REPS_PER_SET = 12
CONFIG_MAX_WEIGHT = 200

#-----------------------------------------------------------------
#assuming they do 25 sets per workout, everyday
#its 40 days before there are 1000 sets
#and after 1000 timesteps the LSTM doesnt really remember further
CONFIG_DAYS_SINCE_LAST_WORKOUT_CAP = 14
#-----------------------------------------------------------------
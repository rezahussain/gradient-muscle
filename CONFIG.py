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


CONFIG_NUM_EPOCHS = 100

CONFIG_SAVE_MODEL_LOCATION = "/Users/admin/Desktop/tmp/stressmodel"

CONFIG_RAW_JSON_PATH = "/Users/admin/PycharmProjects/rezapycharmprojects/gradient-muscle/workoutdata/rezadata/"

#need to make these next three
CONFIG_NN_HUMAN_PICKLES_PATH = "/Users/admin/Desktop/tmp/nnpickles_folder/"
CONFIG_NORMALIZE_VALS_PATH = "/Users/admin/Desktop/tmp/normvals"

CONFIG_WORKOUT_LOOKBACK = 1
CONFIG_BATCH_SIZE = 6
CONFIG_MAX_REPS_PER_SET = 15
CONFIG_MAX_WEIGHT = 1000
#assuming they do 25 sets per workout, everyday
#its 40 days before there are 1000 sets
#and after 1000 timesteps the LSTM doesnt really remember further
CONFIG_DAYS_SINCE_LAST_WORKOUT_CAP = 7*6
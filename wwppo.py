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

# contact info:
# https://www.facebook.com/reza.hussain.98
# reza@dormantlabs.com


import copy
import os
import json
import numpy as np
import pickle
import tensorflow as tf
from random import shuffle
import datetime
import calendar
import math
import CONFIG as CONFIG

jsonfilenames = os.listdir(CONFIG.CONFIG_RAW_JSON_PATH)

jsonfilenames.sort()

jsonobjects = []

for i in range(len(jsonfilenames)):
    d = open(CONFIG.CONFIG_RAW_JSON_PATH + jsonfilenames[i])
    o = json.load(d)
    jsonobjects.append(o)

# we build one datapoint for each workoutday
# it consists of
# the workout
# then the last workout
# then all of the day vectors on the side in the range
# so we build a range for each datapoint
# based off of the workout lookback


workout_indexes = []

for i in range(len(jsonobjects)):
    jo = jsonobjects[i]
    # print jo['user_vector']
    if len(jo['workout_vector_arr']) > 0:
        workout_indexes.append(i)

abc = None

workout_ranges = []
for i in range(len(workout_indexes)):
    behind = workout_indexes[:i]
    if len(behind) > CONFIG.CONFIG_WORKOUT_LOOKBACK:
        end = workout_indexes[i]
        start = workout_indexes[i - CONFIG.CONFIG_WORKOUT_LOOKBACK]
        workout_ranges.append([start, end])
        # if i-CONFIG_WORKOUT_LOOKBACK >= 0:
        #    workout_ranges.append([i-CONFIG_WORKOUT_LOOKBACK,i])


# ---------------------------------------------------------->

# DONE---need to build time vocabulary array
# so from 1-12
# 0 to 59
def generate_time_vocabulary():
    times = []
    for i in range(0, 24):
        h = i
        suffix = None
        if h < 12:
            h = h
            suffix = "am"
        else:
            h = h - 12
            suffix = "pm"
        hstr = str(h)
        if len(hstr) < 2:
            hstr = "0" + hstr
        if hstr == "00":
            hstr = "12"

        for yy in range(0, 60):
            m = yy
            mstr = str(m)
            if len(mstr) < 2:
                mstr = "0" + mstr
            times.append(hstr + ":" + mstr + suffix)
    return times


time_vocabulary = generate_time_vocabulary()

time_vocabulary.append(-1)
# print time_vocabulary

# ---------------------------------------------------------->
# DONE---need to build a vocabulary for the pulled muscle name

pulled_muscle_vocabulary = []
for i in workout_indexes:
    jo = jsonobjects[i]
    workout_vector = jo["workout_vector_arr"]
    for set in workout_vector:
        if set["pulled_muscle_name"] not in pulled_muscle_vocabulary:
            pulled_muscle_vocabulary.append(set["pulled_muscle_name"])

# ---------------------------------------------------------->
# need to build exercise name vocabulary

exercise_vocabulary = []
for i in workout_indexes:
    jo = jsonobjects[i]
    workout_vector = jo["workout_vector_arr"]
    for set in workout_vector:
        if set["exercise_name"] not in exercise_vocabulary:
            exercise_vocabulary.append(set["exercise_name"])

# ---------------------------------------------------------->

# need to find max workout_arr length so we can pad
# all of the workouts to this length
# so we can train in batches
# bc batches need the same shapes

max_workout_range_array_len = 0

for r in workout_ranges:
    workout_range_max_num_sets = 0
    for xx in range(r[0], r[1] + 1):
        workout_range_max_num_sets += len(jsonobjects[xx]["workout_vector_arr"])
    if workout_range_max_num_sets > max_workout_range_array_len:
        max_workout_range_array_len = workout_range_max_num_sets

Abc = None

# ---------------------------------------------------------->

# now we need to find max range day length so we can pad
# all of the day series units to this length
# again so we can train in batches
# bc batches need the same shape

max_day_range_array_len = 0
for r in workout_ranges:
    rdays = r[1] - r[0]
    if rdays > max_day_range_array_len:
        max_day_range_array_len = rdays

# dk if this is a hack
# bc it says 10 but then 11 shows up in the dataset
max_day_range_array_len += 1

Abc = None

# ---------------------------------------------------------->

# weight is in lbs
CHOOSABLE_EXERCISES = ["squat", "benchpress", "deadlift"]
CHOOSABLE_MULTIPLIERS = [-0.50,-0.25,-0.12,-0.05,0.50,0.25,0.12,0.05]
# now we need to make all of the combos the RLAgent can pick
rl_all_possible_actions = []
for exercise_name in CHOOSABLE_EXERCISES:
    exercise_index = exercise_vocabulary.index(exercise_name)
    for x in range(1, CONFIG.CONFIG_MAX_REPS_PER_SET + 1):
        #for y in range(45, CONFIG.CONFIG_MAX_WEIGHT, 5):
        for y in CHOOSABLE_MULTIPLIERS:
            rl_all_possible_actions.append("exercise=" + exercise_name + ":reps=" + str(x) + ":multiplier=" + str(y))

rl_all_possible_actions.append("exercise=LEAVEGYM:reps=0:multiplier=0")

ABC = None

# old was 1356 for 200lb
# new is 451 for 10 multipliers
# the reward that the RL was able to get doubled from ~51 to 108
# when I switched over to less RL actions
# this means the RL has a hard time learning
# when there are many choices


# ---------------------------------------------------------->


def calc_days_since_last_workout(current_workout_yyyymmdd, last_workout_yyyymmdd):
    cwyyyy = int(current_workout_yyyymmdd[0:4])
    cwmm = int(current_workout_yyyymmdd[4:6])
    cwdd = int(current_workout_yyyymmdd[6:8])

    lwyyyy = int(last_workout_yyyymmdd[0:4])
    lwmm = int(last_workout_yyyymmdd[4:6])
    lwdd = int(last_workout_yyyymmdd[6:8])

    cw_datetime = datetime.datetime(cwyyyy, cwmm, cwdd)
    lw_datetime = datetime.datetime(lwyyyy, lwmm, lwdd)

    cw_timestamp = calendar.timegm(cw_datetime.timetuple())
    lw_timestamp = calendar.timegm(lw_datetime.timetuple())

    days_since_last_workout = float(cw_timestamp - lw_timestamp) / (60 * 60 * 24)
    if days_since_last_workout > CONFIG.CONFIG_DAYS_SINCE_LAST_WORKOUT_CAP:
        days_since_last_workout = CONFIG.CONFIG_DAYS_SINCE_LAST_WORKOUT_CAP

    return days_since_last_workout


# ---------------------------------------------------------->


def make_workout_step_human(
        exercise_name,
        reps_planned,
        reps_completed,
        weight_lbs,

        postset_heartrate,
        went_to_failure,
        did_pull_muscle,

        used_lifting_gear,
        unit_days_arr_human,
        velocities_m_per_s_arr

):
    packaged_workout = {}

    # use a hot vector for which exercise they r doing
    for iii in range(len(exercise_vocabulary)):
        packaged_workout["category_exercise_name_" + str(iii)] = 0
    ex_name = exercise_name

    if ex_name is not -1:
        en = exercise_vocabulary.index(ex_name)
        packaged_workout["category_exercise_name_" + str(en)] = 1

    packaged_workout["reps_planned"] = reps_planned
    packaged_workout["reps_completed"] = reps_completed
    packaged_workout["weight_lbs"] = weight_lbs

    packaged_workout["postset_heartrate"] = postset_heartrate
    packaged_workout["went_to_failure"] = went_to_failure

    packaged_workout["did_pull_muscle"] = did_pull_muscle

    packaged_workout["used_lifting_gear"] = used_lifting_gear

    packaged_workout["days_since_last_workout"] = unit_days_arr_human[-1]["days_since_last_workout"]

    # pad reps array to a fixed 20 reps
    # that way you can just include a 20 feature array
    # then let the flow go unmessed with

    vmpsa = velocities_m_per_s_arr
    velarr = []
    if vmpsa != -1:
        velarr.extend(vmpsa)

        # if the bar sensor missed reps
        # we pad reps here till we have matching num of
        # completed reps and velocities
        # what will happen if you dont do this is the model
        # will start to predict when the bar sensor misses reps
        # which we dont want

        reps_completed = reps_completed
        while len(velarr) < reps_completed:
            velarr.append(np.mean(vmpsa))

    while len(velarr) < CONFIG.CONFIG_MAX_REPS_PER_SET:
        velarr.append(0)

    if len(velarr) > CONFIG.CONFIG_MAX_REPS_PER_SET:
        assert "too many velocities, bad data?"

    for c in velarr:
        if c < 0.0:
            assert "there exists a negative velocity, bad data?"

    for iiii in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
        packaged_workout["velocities_arr_" + str(iiii)] = velarr[iiii]

    return packaged_workout


# ---------------------------------------------------------->



print workout_ranges


def make_raw_units():
    for r in workout_ranges:

        # make a unit for each range
        unit = {}
        unit_user = {}
        unit_days = []
        unit_workouts = []
        unit_y = []

        per_set_workout = []

        last_day_vector_workout_day_index = None

        for xx in range(r[0], r[1] + 1):

            debug_name = jsonfilenames[xx]
            # build day array
            # print xx
            packaged_day = {}
            packaged_day["heart_rate_variability_rmssd"] = jsonobjects[xx]["day_vector"]["heart_rate_variability_rmssd"]
            packaged_day["post_day_wearable_calories_burned"] = jsonobjects[xx]["day_vector"][
                "post_day_wearable_calories_burned"]
            packaged_day["post_day_calories_in"] = jsonobjects[xx]["day_vector"]["post_day_calories_in"]
            packaged_day["post_day_protein_g"] = jsonobjects[xx]["day_vector"]["post_day_protein_g"]
            packaged_day["post_day_carbs_g"] = jsonobjects[xx]["day_vector"]["post_day_carbs_g"]
            packaged_day["post_day_fat_g"] = jsonobjects[xx]["day_vector"]["post_day_fat_g"]
            packaged_day["withings_weight_lbs"] = jsonobjects[xx]["day_vector"]["withings_weight_lbs"]
            packaged_day["withings_body_fat_percent"] = jsonobjects[xx]["day_vector"]["withings_body_fat_percent"]
            packaged_day["withings_muscle_mass_percent"] = jsonobjects[xx]["day_vector"]["withings_muscle_mass_percent"]
            packaged_day["withings_body_water_percent"] = jsonobjects[xx]["day_vector"]["withings_body_water_percent"]
            packaged_day["withings_heart_rate_bpm"] = jsonobjects[xx]["day_vector"]["withings_heart_rate_bpm"]
            packaged_day["withings_bone_mass_percent"] = jsonobjects[xx]["day_vector"]["withings_bone_mass_percent"]
            packaged_day["withings_pulse_wave_velocity_m_per_s"] = jsonobjects[xx]["day_vector"][
                "withings_pulse_wave_velocity_m_per_s"]

            sbtap = time_vocabulary.index(jsonobjects[xx]["day_vector"]["sleeptime_bed_time_ampm"])
            packaged_day["sleeptime_bed_time_ampm_index"] = sbtap

            srtap = time_vocabulary.index(jsonobjects[xx]["day_vector"]["sleeptime_rise_time_ampm"])
            packaged_day["sleeptime_rise_time_ampm_index"] = srtap

            packaged_day["sleeptime_efficiency_percent"] = jsonobjects[xx]["day_vector"]["sleeptime_efficiency_percent"]

            sarap = time_vocabulary.index(jsonobjects[xx]["day_vector"]["sleeptime_alarm_ring_ampm"])
            packaged_day["sleeptime_alarm_ring_ampm_index"] = sarap

            sasap = time_vocabulary.index(jsonobjects[xx]["day_vector"]["sleeptime_alarm_set_ampm"])
            packaged_day["sleeptime_alarm_set_ampm_index"] = sasap

            packaged_day["sleeptime_snoozed"] = jsonobjects[xx]["day_vector"]["sleeptime_snoozed"]

            def getNumericalHour(hour_string):
                result = None
                if hour_string != -1:
                    hours = hour_string.split(":")[0]
                    minutes = hour_string.split(":")[1]
                    result = float(hours) + (float(minutes) / 60.0)
                else:
                    result = -1
                return result

            sahrs = getNumericalHour(jsonobjects[xx]["day_vector"]["sleeptime_awake_hrs"])
            packaged_day["sleeptime_awake_hrs"] = sahrs

            slshrs = getNumericalHour(jsonobjects[xx]["day_vector"]["sleeptime_light_sleep_hrs"])
            packaged_day["sleeptime_light_sleep_hrs"] = slshrs

            sdrhrs = getNumericalHour(jsonobjects[xx]["day_vector"]["sleeptime_deep_rem_hrs"])
            packaged_day["sleeptime_deep_rem_hrs"] = sdrhrs

            days_since_last_workout = None
            if last_day_vector_workout_day_index is not None:
                current_workout_yyyymmdd = jsonobjects[xx]["day_vector"]["date_yyyymmdd"]
                last_workout_yyyymmdd = jsonobjects[last_day_vector_workout_day_index]["day_vector"]["date_yyyymmdd"]
                days_since_last_workout = calc_days_since_last_workout(current_workout_yyyymmdd, last_workout_yyyymmdd)
            else:
                days_since_last_workout = 0
            packaged_day["days_since_last_workout"] = days_since_last_workout
            abc = len(packaged_day)
            if len(jsonobjects[xx]["workout_vector_arr"]) > 0:
                last_day_vector_workout_day_index = xx

            # ------------------------------------------------------------

            # if you modify the setup above make sure it is also modded
            # in the pad unit_days below
            unit_days.append(packaged_day)

            # need to do it as one big one
            # bc its easier to make sure that you do not include
            # days that are ahead of the current work set
            # but then still include days that are behind or at the current
            # work set


            if len(jsonobjects[xx]["workout_vector_arr"]) > 0:

                for ii in range(len(jsonobjects[xx]["workout_vector_arr"])):

                    exercise_name = jsonobjects[xx]["workout_vector_arr"][ii]["exercise_name"]
                    reps_planned = jsonobjects[xx]["workout_vector_arr"][ii]["reps_planned"]
                    reps_completed = jsonobjects[xx]["workout_vector_arr"][ii]["reps_completed"]
                    weight_lbs = jsonobjects[xx]["workout_vector_arr"][ii]["weight_lbs"]

                    postset_heartrate = jsonobjects[xx]["workout_vector_arr"][ii]["postset_heartrate"]
                    went_to_failure = jsonobjects[xx]["workout_vector_arr"][ii]["went_to_failure"]
                    did_pull_muscle = jsonobjects[xx]["workout_vector_arr"][ii]["did_pull_muscle"]

                    used_lifting_gear = jsonobjects[xx]["workout_vector_arr"][ii]["used_lifting_gear"]
                    vmpsa = jsonobjects[xx]["workout_vector_arr"][ii]["velocities_m_per_s_arr"]

                    # ------------------------------------------------------------------------


                    packaged_workout = make_workout_step_human(
                        exercise_name,
                        reps_planned,
                        reps_completed,
                        weight_lbs,
                        postset_heartrate,
                        went_to_failure,
                        did_pull_muscle,
                        used_lifting_gear,
                        unit_days,
                        vmpsa
                    )
                    unit_workouts.append(packaged_workout)

                    # ------------------------------------------------------------------------

                    # so basically for each set in the workout_vector array
                    # we make a train unit with x and y
                    # in this we hollow out the x
                    # then use the unhollowed out x for the y
                    # so the model can learn to predict the values we hollowed out

                    unit_workout_clone = unit_workouts[:]

                    original = unit_workout_clone[-1]
                    copyx = copy.deepcopy(original)
                    # copyx = unit_workout_clone[-1][:]
                    copyy = copy.deepcopy(copyx)

                    # exercise_name categories
                    # reps
                    # weight_lbs
                    copyx["reps_completed"] = -1  # reps_completed

                    copyx["postset_heartrate"] = -1  # postset_heartrate
                    copyx["went_to_failure"] = -1  # went to failure
                    copyx["did_pull_muscle"] = -1

                    # for iii in range(len(pulled_muscle_vocabulary)):
                    #    copyx["category_pulled_muscle_"+str(iii)] = 0

                    # used_lifting_gear
                    # dayssincelastworkout

                    # init the reps speeds to 0
                    for iiii in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
                        copyx["velocities_arr_" + str(iiii)] = 0

                    unit_workout_clone[-1] = copyx

                    # ---------------------------------------------------------------------

                    # now pad the unit_workout_timeseries to the max range
                    # we have seen, so make it the length of the workout with the max number
                    # of sets that exists in our whole dataset
                    # this makes it so we can train
                    # batches bc for batches the inputs all have to have
                    # the same shape for some reason

                    unit_workout_clone_padded = unit_workout_clone[:]

                    padx = copy.deepcopy(copyx)
                    # exercise_name index
                    for iii in range(len(exercise_vocabulary)):
                        padx["category_exercise_name_" + str(iii)] = 0
                    padx["reps_planned"] = -1
                    padx["reps_completed"] = -1
                    padx["weight_lbs"] = -1

                    padx["postset_heartrate"] = -1
                    padx["went_to_failure"] = 0
                    padx["did_pull_muscle"] = 0

                    padx["used_lifting_gear"] = 0
                    padx["days_since_last_workout"] = 0
                    # init the reps speeds to 0
                    for iiii in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
                        padx["velocities_arr_" + str(iiii)] = 0

                    while len(unit_workout_clone_padded) < max_workout_range_array_len:
                        unit_workout_clone_padded.insert(0, padx)

                    # ---------------------------------------------------------------------

                    # now pad the day series to the max range
                    # we have seen, so we can train using batches
                    # bc batches need the same size shapes

                    unit_days_padded = unit_days[:]

                    packaged_day_padded = {}
                    packaged_day_padded["heart_rate_variability_rmssd"] = -1
                    packaged_day_padded["post_day_wearable_calories_burned"] = -1
                    packaged_day_padded["post_day_calories_in"] = -1
                    packaged_day_padded["post_day_protein_g"] = -1
                    packaged_day_padded["post_day_carbs_g"] = -1
                    packaged_day_padded["post_day_fat_g"] = -1
                    packaged_day_padded["withings_weight_lbs"] = -1
                    packaged_day_padded["withings_body_fat_percent"] = -1
                    packaged_day_padded["withings_muscle_mass_percent"] = -1
                    packaged_day_padded["withings_body_water_percent"] = -1
                    packaged_day_padded["withings_heart_rate_bpm"] = -1
                    packaged_day_padded["withings_bone_mass_percent"] = -1
                    packaged_day_padded["withings_pulse_wave_velocity_m_per_s"] = -1
                    packaged_day_padded["sleeptime_bed_time_ampm_index"] = -1
                    packaged_day_padded["sleeptime_rise_time_ampm_index"] = -1
                    packaged_day_padded["sleeptime_efficiency_percent"] = -1

                    packaged_day_padded["sleeptime_alarm_ring_ampm_index"] = -1
                    packaged_day_padded["sleeptime_alarm_set_ampm_index"] = -1
                    packaged_day_padded["sleeptime_snoozed"] = -1

                    packaged_day_padded["sleeptime_awake_hrs"] = -1
                    packaged_day_padded["sleeptime_light_sleep_hrs"] = -1
                    packaged_day_padded["sleeptime_deep_rem_hrs"] = -1
                    packaged_day_padded["days_since_last_workout"] = -1

                    while len(unit_days_padded) < max_day_range_array_len:
                        unit_days_padded.insert(0, packaged_day_padded)

                    ABC = len(unit_days_padded)
                    DEF = None

                    # ---------------------------------------------------------------------


                    # only want to train it with at least one of
                    # the measured exertions
                    # heartrate or speed




                    has_heartrate = True
                    if (
                                copyy["postset_heartrate"] == -1

                    ):
                        has_heartrate = False

                    has_velocities = False
                    for iiii in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
                        if copyy["velocities_arr_" + str(iiii)] > 0.0:
                            has_velocities = True

                    has_valid_y = False

                    if has_velocities and has_heartrate:
                        has_valid_y = True

                        # UPDATE: model gets horrible with either
                        # need to use both or one or the other

                        # not sure how I feel about using either
                        # instead of and
                        # but some lifters will not record hr
                        # might have to branch model out
                        # but training it together gives it a better
                        # understanding even with partial data

                    copyyy = {}
                    copyyy["reps_completed"] = copyy["reps_completed"]
                    copyyy["postset_heartrate"] = copyy["postset_heartrate"]
                    copyyy["went_to_failure"] = copyy["went_to_failure"]
                    copyyy["did_pull_muscle"] = copyy["did_pull_muscle"]
                    for iiii in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
                        copyyy["velocities_arr_" + str(iiii)] = copyy["velocities_arr_" + str(iiii)]

                    ABC = None

                    userjson = jsonobjects[xx]["user_vector"]
                    userx = {}
                    userx["genetically_gifted"] = userjson["genetically_gifted"]
                    userx["years_old"] = userjson["years_old"]
                    userx["wrist_width_inches"] = userjson["wrist_width_inches"]
                    userx["ankle_width_inches"] = userjson["ankle_width_inches"]
                    userx["sex_is_male"] = userjson["sex_is_male"]
                    userx["height_inches"] = userjson["height_inches"]
                    userx["harris_benedict_bmr"] = userjson["harris_benedict_bmr"]

                    # this is what we r gonna package for the NN
                    workoutxseries = unit_workout_clone_padded[:]
                    workouty = copy.deepcopy(copyyy)
                    dayseries = unit_days_padded[:]
                    userx = userx

                    wholeTrainUnit = {}
                    wholeTrainUnit["workoutxseries"] = workoutxseries
                    wholeTrainUnit["workouty"] = workouty
                    wholeTrainUnit["dayseriesx"] = dayseries
                    wholeTrainUnit["userx"] = userx

                    savename = jsonobjects[xx]["day_vector"]["date_yyyymmdd"] + "_" + str(ii)

                    if has_valid_y:
                        pickle.dump(wholeTrainUnit, open(CONFIG.CONFIG_NN_HUMAN_PICKLES_PATH + savename, "wb"))


def get_raw_pickle_filenames():
    picklefilenames = os.listdir(CONFIG.CONFIG_NN_HUMAN_PICKLES_PATH)
    if ".DS_Store" in picklefilenames:
        picklefilenames.remove(".DS_Store")
    return picklefilenames


def get_norm_values():
    normVals = pickle.load(open(CONFIG.CONFIG_NORMALIZE_VALS_PATH, "rb"))
    return normVals


def write_norm_values():
    picklefilenames = get_raw_pickle_filenames()

    unpickled_package = pickle.load(open(CONFIG.CONFIG_NN_HUMAN_PICKLES_PATH + picklefilenames[0], "rb"))

    dayseriesxmin = [9999.0] * len((unpickled_package["dayseriesx"][0]).keys())
    dayseriesxmax = [-9999.0] * len((unpickled_package["dayseriesx"][0]).keys())

    userxmin = [9999.0] * len((unpickled_package["userx"]).keys())
    userxmax = [-9999.0] * len((unpickled_package["userx"]).keys())

    workoutxseriesmin = [9999.0] * len((unpickled_package["workoutxseries"][0]).keys())
    workoutxseriesmax = [-9999.0] * len((unpickled_package["workoutxseries"][0]).keys())

    workoutymin = [9999.0] * len((unpickled_package["workouty"]).keys())
    workoutymax = [-9999.0] * len((unpickled_package["workouty"]).keys())

    for picklefilename in picklefilenames:

        unpickled_package = pickle.load(open(CONFIG.CONFIG_NN_HUMAN_PICKLES_PATH + picklefilename, "rb"))

        unpickledayseriesx = unpickled_package["dayseriesx"]
        for i in range(len(unpickledayseriesx)):
            daystep = unpickledayseriesx[i]
            daystepkeys = sorted(list(daystep.keys()))
            for ii in range(len(daystepkeys)):
                akey = daystepkeys[ii]
                aval = float(daystep[akey])
                if aval < dayseriesxmin[ii]:
                    dayseriesxmin[ii] = aval
                if aval > dayseriesxmax[ii]:
                    dayseriesxmax[ii] = aval

        unpickleduserx = unpickled_package["userx"]
        userxkeys = sorted(list(unpickleduserx.keys()))
        for i in range(len(userxkeys)):
            akey = userxkeys[i]
            aval = float(unpickleduserx[akey])
            if aval < userxmin[i]:
                userxmin[i] = aval
            if aval > userxmax[i]:
                userxmax[i] = aval

        unpickledworkoutxseries = unpickled_package["workoutxseries"]
        for i in range(len(unpickledworkoutxseries)):
            workoutstep = unpickledworkoutxseries[i]
            workoutstepkeys = sorted(list(workoutstep.keys()))
            for ii in range(len(workoutstepkeys)):
                akey = workoutstepkeys[ii]
                aval = float(workoutstep[akey])
                if aval < workoutxseriesmin[ii]:
                    workoutxseriesmin[ii] = aval
                if aval > workoutxseriesmax[ii]:
                    workoutxseriesmax[ii] = aval

        unpickledworkouty = unpickled_package["workouty"]
        unpickledworkoutykeys = sorted(list(unpickledworkouty.keys()))
        for i in range(len(unpickledworkoutykeys)):
            akey = unpickledworkoutykeys[i]
            aval = float(unpickledworkouty[akey])
            if aval < workoutymin[i]:
                workoutymin[i] = aval
            if aval > workoutymax[i]:
                workoutymax[i] = aval

    normVals = {}
    normVals["dayseriesxmin"] = dayseriesxmin
    normVals["daysseriesxmax"] = dayseriesxmax
    normVals["userxmin"] = userxmin
    normVals["userxmax"] = userxmax
    normVals["workoutxseriesmin"] = workoutxseriesmin
    normVals["workoutxseriesmax"] = workoutxseriesmax
    normVals["workoutymin"] = workoutymin
    normVals["workoutymax"] = workoutymax

    pickle.dump(normVals, open(CONFIG.CONFIG_NORMALIZE_VALS_PATH, "wb"))


def normalize_unit(packaged_unit, norm_vals):
    dayseriesxmin = norm_vals["dayseriesxmin"]
    dayseriesxmax = norm_vals["daysseriesxmax"]
    userxmin = norm_vals["userxmin"]
    userxmax = norm_vals["userxmax"]
    workoutxseriesmin = norm_vals["workoutxseriesmin"]
    workoutxseriesmax = norm_vals["workoutxseriesmax"]
    workoutymin = norm_vals["workoutymin"]
    workoutymax = norm_vals["workoutymax"]

    n_packaged_unit = {}

    unpickledayseriesx = packaged_unit["dayseriesx"]
    n_unpickleddayseriesx = []
    for i in range(len(unpickledayseriesx)):
        daystep = unpickledayseriesx[i]
        ndaystep = []
        daystepkeys = sorted(list(daystep.keys()))
        for ii in range(len(daystepkeys)):
            akey = daystepkeys[ii]
            aval = daystep[akey]
            if aval > dayseriesxmax[ii]:
                aval = dayseriesxmax[ii]
            if aval < dayseriesxmin[ii]:
                aval = dayseriesxmin[ii]
            if (dayseriesxmax[ii] - dayseriesxmin[ii]) > 0:
                aval = (aval - dayseriesxmin[ii]) / (dayseriesxmax[ii] - dayseriesxmin[ii])
            else:
                aval = 0
            ndaystep.append(aval)
        n_unpickleddayseriesx.append(ndaystep)
    n_packaged_unit["dayseriesx"] = n_unpickleddayseriesx

    unpickledworkoutxseries = packaged_unit["workoutxseries"]
    n_unpickledworkoutxseries = []
    for i in range(len(unpickledworkoutxseries)):
        workoutstep = unpickledworkoutxseries[i]
        nworkoutstep = []
        workoutstepkeys = sorted(list(workoutstep.keys()))
        for ii in range(len(workoutstepkeys)):
            akey = workoutstepkeys[ii]
            aval = workoutstep[akey]
            if aval > workoutxseriesmax[ii]:
                aval = workoutxseriesmax[ii]
            if aval < workoutxseriesmin[ii]:
                aval = workoutxseriesmin[ii]
            if (workoutxseriesmax[ii] - workoutxseriesmin[ii]) > 0:
                aval = (aval - workoutxseriesmin[ii]) / (workoutxseriesmax[ii] - workoutxseriesmin[ii])
            else:
                aval = 0
            nworkoutstep.append(aval)
        n_unpickledworkoutxseries.append(nworkoutstep)
    n_packaged_unit["workoutxseries"] = n_unpickledworkoutxseries

    unpickleduserx = packaged_unit["userx"]
    n_unpickleduserx = []
    unpickleduserxkeys = sorted(list(unpickleduserx.keys()))
    for i in range(len(unpickleduserxkeys)):
        akey = unpickleduserxkeys[i]
        aval = unpickleduserx[akey]
        if aval > userxmax[i]:
            aval = userxmax[i]
        if aval < userxmin[i]:
            aval = userxmin[i]
        if (userxmax[i] - userxmin[i]) > 0:
            aval = (aval - userxmin[i]) / (userxmax[i] - userxmin[i])
        else:
            aval = 0
        n_unpickleduserx.append(aval)
    n_packaged_unit["userx"] = n_unpickleduserx

    unpickledworkouty = packaged_unit["workouty"]
    n_unpickledworkouty = []
    unpickledworkoutykeys = sorted(list(unpickledworkouty.keys()))
    for i in range(len(unpickledworkoutykeys)):
        akey = unpickledworkoutykeys[i]
        aval = unpickledworkouty[akey]
        if aval > workoutymax[i]:
            aval = workoutymax[i]
        if aval < workoutymin[i]:
            aval = workoutymin[i]
        if (workoutymax[i] - workoutymin[i]) > 0:
            aval = (aval - workoutymin[i]) / (workoutymax[i] - workoutymin[i])
        else:
            aval = 0
        n_unpickledworkouty.append(aval)
    n_packaged_unit["workouty"] = n_unpickledworkouty

    return n_packaged_unit


def make_h_workout_with_xh_ym(workoutstep_xh, workoutstep_ym, days_series_arr_h):
    norm_vals = pickle.load(open(CONFIG.CONFIG_NORMALIZE_VALS_PATH, "rb"))

    dayseriesxmin = norm_vals["dayseriesxmin"]
    dayseriesxmax = norm_vals["daysseriesxmax"]
    userxmin = norm_vals["userxmin"]
    userxmax = norm_vals["userxmax"]
    workoutxseriesmin = norm_vals["workoutxseriesmin"]
    workoutxseriesmax = norm_vals["workoutxseriesmax"]
    workoutymin = norm_vals["workoutymin"]
    workoutymax = norm_vals["workoutymax"]

    workoutstep_hollow_h = copy.deepcopy(workoutstep_xh)

    h_workout_timestep = workoutstep_hollow_h

    ykeys = ["reps_completed", "postset_heartrate", "went_to_failure", "did_pull_muscle"]
    # init the reps speeds to 0
    for iiii in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
        ykeys.append("velocities_arr_" + str(iiii))
    workoutstepkeys = sorted(list(ykeys))

    for ii in range(len(workoutstepkeys)):
        akey = workoutstepkeys[ii]
        aval = workoutstep_ym[ii]
        amin = workoutymin[ii]
        amax = workoutymax[ii]
        unnormal = (aval * (amax - amin)) + amin
        h_workout_timestep[akey] = unnormal
        ABC = None

    return h_workout_timestep


def denormalize_workout_series_individual_timestep(n_workout_timestep, days_series_arr_h):
    norm_vals = pickle.load(open(CONFIG.CONFIG_NORMALIZE_VALS_PATH, "rb"))

    dayseriesxmin = norm_vals["dayseriesxmin"]
    dayseriesxmax = norm_vals["daysseriesxmax"]
    userxmin = norm_vals["userxmin"]
    userxmax = norm_vals["userxmax"]
    workoutxseriesmin = norm_vals["workoutxseriesmin"]
    workoutxseriesmax = norm_vals["workoutxseriesmax"]
    workoutymin = norm_vals["workoutymin"]
    workoutymax = norm_vals["workoutymax"]

    # first make a hollow object
    exercise_name = -1
    reps_planned = -1
    reps_completed = -1
    weight_lbs = -1

    postset_heartrate = -1
    went_to_failure = -1
    did_pull_muscle = -1

    used_lifting_gear = -1
    unit_days_arr_human = days_series_arr_h
    velocities_m_per_s_arr = -1

    workoutstep_hollow_h = make_workout_step_human(
        exercise_name,
        reps_planned,
        reps_completed,
        weight_lbs,

        postset_heartrate,
        went_to_failure,
        did_pull_muscle,

        used_lifting_gear,
        unit_days_arr_human,
        velocities_m_per_s_arr
    )

    h_workout_timestep = {}
    workoutstepkeys = sorted(list(workoutstep_hollow_h.keys()))
    for ii in range(len(workoutstepkeys)):
        akey = workoutstepkeys[ii]
        aval = n_workout_timestep[ii]
        amin = workoutxseriesmin[ii]
        amax = workoutxseriesmax[ii]
        unnormal = (aval * (amax - amin)) + amin
        h_workout_timestep[akey] = unnormal
        ABC = None

    return h_workout_timestep


def get_unit_names():
    picklefilenames = os.listdir(CONFIG.CONFIG_NN_HUMAN_PICKLES_PATH)
    if ".DS_Store" in picklefilenames:
        picklefilenames.remove(".DS_Store")
    return picklefilenames


def get_human_unit_for_name(unit_name):
    unpickled_package = pickle.load(open(CONFIG.CONFIG_NN_HUMAN_PICKLES_PATH + unit_name, "rb"))
    return unpickled_package


def get_machine_unit_for_name(unit_name, norm_vals):
    human_unit = get_human_unit_for_name(unit_name)
    machine_unit = normalize_unit(human_unit, norm_vals)
    return machine_unit


def convert_human_unit_to_machine(h_unit, norm_vals):
    m_unit = normalize_unit(h_unit, norm_vals)
    return m_unit


# a_unit = getUnitForName(all_names[0])

class Lift_NN():
    def __init__(self, a_dayseriesx, a_userx, a_workoutxseries, a_workouty, CHOSEN_BATCH_SIZE):
        # --------------------------------------------------------------------------------------------------------------
        with tf.variable_scope('stress_model'):
            self.WORLD_NUM_Y_OUTPUT = len(a_workouty)

            self.world_day_series_input = tf.placeholder(tf.float32, (None, None, len(a_dayseriesx[0])),
                                                         name="world_day_series_input")
            self.world_workout_series_input = tf.placeholder(tf.float32, (None, None, len(a_workoutxseries[0])),
                                                             name="world_workout_series_input")

            self.world_user_vector_input = tf.placeholder(tf.float32, (None, len(a_userx)),
                                                          name="world_user_vector_input")
            self.world_workout_y = tf.placeholder(tf.float32, (None, len(a_workouty)), name="world_workouty")

            with tf.variable_scope('world_workout_series_stageA'):
                world_wo_cellA = tf.contrib.rnn.LSTMCell(100)
                world_wo_rnn_outputsA, world_wo_rnn_stateA = tf.nn.dynamic_rnn(world_wo_cellA,
                                                                               self.world_workout_series_input,
                                                                               dtype=tf.float32)
                world_wo_batchA = tf.layers.batch_normalization(world_wo_rnn_outputsA)

            with tf.variable_scope('world_workout_series_stageB'):
                world_wo_cellB = tf.contrib.rnn.LSTMCell(100)
                world_wo_resB = tf.contrib.rnn.ResidualWrapper(world_wo_cellB)
                world_wo_rnn_outputsB, world_wo_rnn_stateB = tf.nn.dynamic_rnn(world_wo_resB, world_wo_rnn_outputsA,
                                                                               dtype=tf.float32)
                world_wo_batchB = tf.layers.batch_normalization(world_wo_rnn_outputsB)

            with tf.variable_scope('world_workout_series_stageC'):
                world_wo_cellC = tf.contrib.rnn.LSTMCell(100)
                world_wo_resC = tf.contrib.rnn.ResidualWrapper(world_wo_cellC)
                world_wo_rnn_outputsC, world_wo_rnn_stateB = tf.nn.dynamic_rnn(world_wo_resC, world_wo_rnn_outputsB,
                                                                               dtype=tf.float32)
                world_wo_batchC = tf.layers.batch_normalization(world_wo_rnn_outputsC)


            with tf.variable_scope('world_day_series_stageA'):
                world_day_cellA = tf.contrib.rnn.LSTMCell(100)
                world_day_rnn_outputsA, world_day_rnn_stateA = tf.nn.dynamic_rnn(world_day_cellA,
                                                                                 self.world_day_series_input,
                                                                                 dtype=tf.float32)
                world_day_batchA = tf.layers.batch_normalization(world_day_rnn_outputsA)

            with tf.variable_scope('world_day_series_stageB'):
                world_day_cellB = tf.contrib.rnn.LSTMCell(100)
                world_day_resB = tf.contrib.rnn.ResidualWrapper(world_day_cellB)
                world_day_rnn_outputsB, world_day_rnn_stateB = tf.nn.dynamic_rnn(world_day_resB, world_day_rnn_outputsA,
                                                                                 dtype=tf.float32)
                world_day_batchB = tf.layers.batch_normalization(world_day_rnn_outputsB)

            with tf.variable_scope('world_day_series_stageC'):
                world_day_cellC = tf.contrib.rnn.LSTMCell(100)
                world_day_resC = tf.contrib.rnn.ResidualWrapper(world_day_cellC)
                world_day_rnn_outputsC, world_day_rnn_stateC = tf.nn.dynamic_rnn(world_day_resC, world_day_rnn_outputsB,
                                                                                 dtype=tf.float32)
                world_day_batchC = tf.layers.batch_normalization(world_day_rnn_outputsC)



            '''
            with tf.variable_scope('workout_input'):
                world_cellA = tf.contrib.rnn.NASCell(50)
                world_rnn_outputsA, world_rnn_stateA = tf.nn.dynamic_rnn(world_cellA, self.world_workout_series_input, dtype=tf.float32)

            with tf.variable_scope('day_input'):
                world_cellAA = tf.contrib.rnn.NASCell(50)
                world_rnn_outputsAA, world_rnn_stateAA = tf.nn.dynamic_rnn(world_cellAA,  self.world_day_series_input, dtype=tf.float32)

            '''

            '''
            with tf.variable_scope('world_workout_series_stageA'):
                world_cellA = tf.contrib.rnn.LSTMCell(1000)
                world_rnn_outputsA, world_rnn_stateA = tf.nn.dynamic_rnn(world_cellA, self.world_workout_series_input, dtype=tf.float32)

            with tf.variable_scope('world_workout_series_stageB'):
                world_cellB = tf.contrib.rnn.LSTMCell(1000)
                world_rnn_outputsB, world_rnn_stateB = tf.nn.dynamic_rnn(world_cellB, world_rnn_outputsA, dtype=tf.float32)

            with tf.variable_scope('world_day_series_stageA'):
                world_cellAA = tf.contrib.rnn.LSTMCell(1000)
                world_rnn_outputsAA, world_rnn_stateAA = tf.nn.dynamic_rnn(world_cellAA, self.world_day_series_input, dtype=tf.float32)

            with tf.variable_scope('world_day_series_stageB'):
                world_cellBB = tf.contrib.rnn.LSTMCell(1000)
                world_rnn_outputsBB, world_rnn_stateBB = tf.nn.dynamic_rnn(world_cellBB, world_rnn_outputsAA, dtype=tf.float32)
            '''

            world_lastA = world_wo_rnn_outputsC[:, -1:]  # get last lstm output
            world_lastAA = world_day_rnn_outputsC[:, -1:]  # get last lstm output

            # world_lastA = world_wo_batchB[:, -1:]  # get last lstm output
            # world_lastAA = world_day_batchB[:, -1:]  # get last lstm output


            # world_lastA = world_wo_rnn_outputsA[:, -1:]  # get last lstm output
            # world_lastAA = world_day_rnn_outputsAA[:, -1:]  # get last lstm output

            self.world_lastA = world_lastA
            self.world_lastAA = world_lastAA

            # takes those two 250 and concats them to a 500
            self.world_combined = tf.concat([world_lastA, world_lastAA], 2)
            self.world_b4shape = tf.shape(self.world_combined)

            # so at setup time you need to know the shape
            # otherwise it is none
            # and the dense layer cannot be setup with a none dimension
            self.world_combined_shaped = tf.reshape(self.world_combined, (CHOSEN_BATCH_SIZE, 100 + 100))
            self.world_afshape = tf.shape(self.world_combined_shaped)
            # tf.set_shape()

            self.world_combined2 = tf.concat([self.world_combined_shaped, self.world_user_vector_input], 1)
            world_dd = tf.layers.dense(self.world_combined2, self.WORLD_NUM_Y_OUTPUT)

            self.world_y = world_dd

            self.world_e = tf.losses.mean_squared_error(self.world_workout_y, self.world_y)
            self.world_operation = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.world_e)

        ##--------------------------------------------------------------------------------------------------------------

        ##let the agent pick DONE as an exercise
        ##when the env sees that it moves it to the next day
        ##let the agent pick the exercises too
        # just penalize it if it does a circuit
        # can use the same setup as worldNN but diff output


        self.agent_day_series_input = tf.placeholder(tf.float32, (None, None, len(a_dayseriesx[0])),
                                                     name="agent_day_series_input")

        self.agent_workout_series_input = tf.placeholder(tf.float32, (None, None, len(a_workoutxseries[0])),
                                                         name="agent_workout_series_input")

        self.agent_user_vector_input = tf.placeholder(tf.float32, (None, len(a_userx)), name="agent_user_vector_input")



        def add_rl_agent_with_scope(chosen_scope,a_workout_series_input,a_day_series_input,a_user_vector_input):

            with tf.variable_scope(chosen_scope):

                #DAYSERIESWINDOWSIZE = 11
                #WORKOUTSERIESWINDOWSIZE = 32
                ##--------change the below world to agent
                ##--------then setup the outputs
                ##do rl assembling of inputs later when u start coding the rl environment interaction code

                AGENT_NUM_Y_OUTPUT = len(rl_all_possible_actions)
                NUM_NEURONS = 250

                with tf.variable_scope('agent_workout_series_stage_A'+chosen_scope):
                    agent_cellA = tf.contrib.rnn.LSTMCell(250)
                    agent_rnn_outputsA, agent_rnn_stateA = tf.nn.dynamic_rnn(agent_cellA, a_workout_series_input,
                                                                             dtype=tf.float32)

                '''
                with tf.variable_scope('agent_workout_series_stage_B'+chosen_scope):
                    agent_cellB = tf.contrib.rnn.LSTMCell(250)
                    res_agent_cellB = tf.contrib.rnn.ResidualWrapper(agent_cellB)
                    agent_rnn_outputsB, agent_rnn_stateB = tf.nn.dynamic_rnn(res_agent_cellB, agent_rnn_outputsA,
                                                                             dtype=tf.float32)
                '''

                with tf.variable_scope('agent_day_series_stage_A'+chosen_scope):
                    agent_cellAA = tf.contrib.rnn.LSTMCell(250)
                    agent_rnn_outputsAA, agent_rnn_stateAA = tf.nn.dynamic_rnn(agent_cellAA, a_day_series_input,
                                                                               dtype=tf.float32)

                '''
                with tf.variable_scope('agent_day_series_stage_B'+chosen_scope):
                    agent_cellBB = tf.contrib.rnn.LSTMCell(250)
                    res_agent_cellBB = tf.contrib.rnn.ResidualWrapper(agent_cellBB)
                    agent_rnn_outputsBB, agent_rnn_stateBB = tf.nn.dynamic_rnn(res_agent_cellBB, agent_rnn_outputsAA,
                                                                               dtype=tf.float32)
                '''

                agent_lastA = agent_rnn_outputsA[:, -1:]  # get last lstm output
                agent_lastAA = agent_rnn_outputsAA[:, -1:]  # get last lstm output

                agent_lastA = agent_lastA
                agent_lastAA = agent_lastAA

                # takes those two 250 and concats them to a 500
                agent_combined = tf.concat([agent_lastA, agent_lastAA], 2)
                agent_b4shape = tf.shape(agent_combined)

                # so at setup time you need to know the shape
                # otherwise it is none
                # and the dense layer cannot be setup with a none dimension

                # self.agent_combined_shaped = tf.reshape(self.agent_combined,(1,500))

                # self.agent_b4shape can be (10,1,500) when doing rl gradient calcs
                # so in that case we pass the batch num to it
                # so the end result becomes 10,500
                # when we r doing rl agent decision making we use a batch of 1
                # so in that case it looks like 1,500

                agent_combined_shaped = tf.reshape(agent_combined, (-1, 500))

                # self.agent_combined_shaped = tf.reshape(self.agent_combined, [-1])

                agent_afshape = tf.shape(agent_combined_shaped)

                agent_combined2 = tf.concat([agent_combined_shaped, a_user_vector_input], 1)

                agent_dd = tf.layers.dense(agent_combined2, AGENT_NUM_Y_OUTPUT)

                dd3 = tf.contrib.layers.softmax(agent_dd)

                agent_y_policy = dd3

                agent_value = tf.layers.dense(agent_combined2, 1)

            return agent_y_policy,agent_value


        agent_policy1,agent_value1 = add_rl_agent_with_scope("rl_agent1",
                                                             self.agent_workout_series_input,
                                                             self.agent_day_series_input,
                                                             self.agent_user_vector_input)

        agent_policy2,agent_value2 = add_rl_agent_with_scope("rl_agent2",
                                                             self.agent_workout_series_input,
                                                             self.agent_day_series_input,
                                                             self.agent_user_vector_input)

        self.agent_policy1 = agent_policy1
        self.agent_value1 = agent_value1

        #--------------------------------------------------------------------------------------------------------------

        # RL setup is actor critic
        # so we calc a value function of how good it is to be in a certain state
        # then also calculate a policy
        # when updating the gradients for the policy, we take into account what the value function said

        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32, name="reward_holder")
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32, name="action_holder")
        self.value_holder = tf.placeholder(shape=[None], dtype=tf.float32, name="value_holder")
        self.advantage_holder = tf.placeholder(shape=[None], dtype=tf.float32, name="advantage_holder")

        # we concatenate all of the actions from each observation
        # this makes the index that each observed action would have in the concatenated array
        # the first part makes the indexes for each 0 action index
        # then adding_the action holder adds the actual action offset to each index of all of the

        # tldr it makes an array of indexes for an array where all of the action arrays are concatenated
        # this gives us the outputs to apply the reward function to

        # aka get the value outputs so we can compare them to the rewards they gave
        # and adjust them

        value_indexes1 = tf.range(0, tf.shape(agent_value1)[0]) * tf.shape(agent_value1)[1]
        responsible_values1 = tf.gather(tf.reshape(agent_value1, [-1]), value_indexes1)
        self.value_loss1 = tf.reduce_sum(tf.squared_difference(self.reward_holder, responsible_values1))

        # do the same for the policy
        # 'advantage' here is defined as how much better or worse the result was from the prediction

        indexes1 = tf.range(0, tf.shape(agent_policy1)[0]) * tf.shape(agent_policy1)[
            1] + self.action_holder
        # -1 bc the y comes out as  [[blah,blah],[blah,blah]], so reshape converts to [blah,blah,blah,blah]
        responsible_outputs1 = tf.gather(tf.reshape(agent_policy1, [-1]), indexes1)

        #-----------

        value_indexes2 = tf.range(0, tf.shape(agent_value2)[0]) * tf.shape(agent_value2)[1]
        responsible_values2 = tf.gather(tf.reshape(agent_value2, [-1]), value_indexes2)
        self.value_loss2 = tf.reduce_sum(tf.squared_difference(self.reward_holder, responsible_values2))

        # do the same for the policy
        # 'advantage' here is defined as how much better or worse the result was from the prediction

        indexes2 = tf.range(0, tf.shape(agent_policy2)[0]) * tf.shape(agent_policy2)[
            1] + self.action_holder
        # -1 bc the y comes out as  [[blah,blah],[blah,blah]], so reshape converts to [blah,blah,blah,blah]
        responsible_outputs2 = tf.gather(tf.reshape(agent_policy2, [-1]), indexes2)

        # so if the advantage is positive
        # it means the action is better than what the policy would have chosen
        # in that case we want to multiply the policy by the advantage so that
        # it is that much more likely to pick the action that gave more advantage
        # if the advantage is negative then it is worse than what the policy would
        # have chosen, so we want to make it less likely to be picked by the policy
        # so we multiply it by the policy

        # we want to increase the mean
        # bc advantages are derived from rewards
        # and the advantage part cannot be adjusted here
        # so it has to adjust the responsible outputs
        # so when the advantage is negative to increase the mean
        # we have to shrink the responsible output
        # which makes it pick negative actions less

        # so when the advantage is positive we want to increase the mean
        # which means increasing the gradients of the responsible output

        self.policy_loss1 = -tf.reduce_sum(tf.log(responsible_outputs1) * self.advantage_holder)

        # we look at the output of the policy, if it had low confidence in its choice
        # like all the choices were rated almost the same number
        # and lets say something bad happened or good happened
        # dont adjust the gradients that much
        #entropy = -tf.reduce_sum(agent_policy1 * tf.log(agent_policy1))

        # so we want to spend some of the step to increase our value estimator
        # bc this helps make a better advantage estimator
        # but not all, we want the focus of the gradients to a better policy
        # so we take value gradients times half
        # I ommitted entropy here
        # the reasoning is that here I have 8000 actions
        # in that one guy's doom example he only had 3 actions
        # the confidence of any one value in the list of 8000 is not gonna be
        # significantly higher than the rest so the entropy value
        # will always signify the model is unconfident
        # so I omit it
        #self.loss = 0.5 * self.value_loss + self.policy_loss  # - entropy * 0.01

        # reduce_mean will become max mean when passed through optimizer minimize
        # which is what we want
        # if the advantage is positive then we need to increase the action probability
        # if the advantage is negative then we need to decrease the action probability

        # the normal definition of ppo does not include a value_loss
        # but I added it in
        # all ppo does is limit to a trust region by clipping if the adjustment is too big
        # so what I did was add the value estimate to it too and clip that too

        # you do -tf.reduce_mean
        # all it does is calc the mean
        # but minimize tries to minimizes this mean
        # so we invert it so minimize maximizes this mean
        # and to do that you have to increase the action probability
        # for actions with high advantage and reduce it for actions with negative
        # advantage

        epsilon = 0.1
        #self.ratio = (responsible_outputs1/responsible_outputs2)
        self.ppoloss = -tf.reduce_mean(tf.minimum(
                    (responsible_outputs1/responsible_outputs2)*self.advantage_holder,
                    tf.clip_by_value((responsible_outputs1/responsible_outputs2),1-epsilon,1+epsilon)*self.advantage_holder
                )) + (tf.clip_by_value(self.value_loss1*0.5,1-epsilon,1+epsilon))

        #self.loss2 = tf.divide(responsible_outputs1,responsible_outputs2)*self.advantage_holder
        #self.loss2 = self.loss

        self.a3closs = 0.5 * self.value_loss1 + self.policy_loss1  # - entropy * 0.01

        self.loss = self.ppoloss
        #self.loss = self.a3closs

        new_params  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rl_agent1')
        old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rl_agent2')
        self.copy_new_to_old = [old_params.assign(new_params) for new_params, old_params in zip(new_params, old_params)]

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        self.train_value_op = optimizer.minimize(self.value_loss1)


        #-----------------------------------------------------------------------------


        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)

        self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='rl_agent1')

        self.gradient_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        #so a note on tf.gradients
        #your loss func needs to touch all of the tvars that you provide
        #if it doesnt then it returns nonetype for the ones
        #it doesnt touch and crashes
        self.gradient = tf.gradients(self.loss, self.tvars)
        var_norms = tf.global_norm(self.tvars)



        # self.grad_n, _ = tf.clip_by_global_norm(self.gradient_holders,clip_norm=10)

        self.grad_n = self.gradient_holders
        # self.grad_n, _ = tf.clip_by_global_norm(self.gradient_holders, 40)
        # self.grad_n = tf.clip_by_norm(self.gradient_holders, 5)

        # self.grad_n, _ = tf.clip_by_global_norm(self.gradient, var_norms)
        # self.grad_n = tf.clip_by_value(self.gradient, -20,20)

        # self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))
        self.update_batch = optimizer.apply_gradients(zip(self.grad_n, self.tvars))

        # ------------------------------------------------------------------------------

        self.asaver = tf.train.Saver()


def build_batch_from_names(batch_unit_names, batch_size, for_human=None):
    assert for_human is not None, "forgot to specify for_human flag build_batch_from_names"

    day_series_batch = []
    user_x_batch = []
    workout_series_batch = []
    workout_y_batch = []

    norm_vals = get_norm_values()
    i = 0
    while i < batch_size:
        i = i + 1
        a_name = batch_unit_names.pop(0)

        loaded_unit = None
        if for_human:
            loaded_unit = get_human_unit_for_name(a_name)
        else:
            loaded_unit = get_machine_unit_for_name(a_name, norm_vals)

        a_day_series = loaded_unit["dayseriesx"]
        a_user_x = loaded_unit["userx"]
        a_workout_series = loaded_unit["workoutxseries"]
        a_workout_y = loaded_unit["workouty"]

        day_series_batch.append(a_day_series)
        user_x_batch.append(a_user_x)
        workout_series_batch.append(a_workout_series)
        workout_y_batch.append(a_workout_y)

    day_series_batch = np.array(day_series_batch)
    user_x_batch = np.array(user_x_batch)
    workout_series_batch = np.array(workout_series_batch)
    workout_y_batch = np.array(workout_y_batch)

    # print workout_y_batch.shape
    # print workout_series_batch.shape
    # print user_x_batch.shape
    # print day_series_batch.shape

    return workout_y_batch, workout_series_batch, user_x_batch, day_series_batch


def train_stress_adaptation_model():
    make_raw_units()
    write_norm_values()

    all_names = get_unit_names()
    norm_vals = get_norm_values()
    loaded_unit = get_machine_unit_for_name(all_names[0], norm_vals)

    some_day_series = loaded_unit["dayseriesx"]
    some_user_x = loaded_unit["userx"]
    some_workout_series = loaded_unit["workoutxseries"]
    some_workout_y = loaded_unit["workouty"]
    abc = None
    alw = Lift_NN(some_day_series, some_user_x, some_workout_series, some_workout_y, CONFIG.CONFIG_BATCH_SIZE)
    init_op = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)

    split_index = int(math.floor(float(len(all_names)) * 0.8))

    shuffle(all_names)
    train_names = all_names[:split_index]
    valid_names = all_names[split_index:]

    best_error = 9999

    for _ in range(0, CONFIG.CONFIG_NUM_EPOCHS):

        train_names_copy = train_names[:]
        shuffle(train_names_copy)

        valid_names_copy = valid_names[:]

        train_error = 0.0
        valid_error = 0.0

        while len(train_names_copy) > CONFIG.CONFIG_BATCH_SIZE:

            batch_unit_train_names = train_names_copy[:CONFIG.CONFIG_BATCH_SIZE]

            for ii in range(CONFIG.CONFIG_BATCH_SIZE):
                train_names_copy.pop(0)

            # day_series_batch = []
            # user_x_batch = []
            # workout_series_batch = []
            # workout_y_batch = []

            wo_y_batch, wo_series_batch, user_x_batch, day_series_batch \
                = build_batch_from_names(batch_unit_train_names, CONFIG.CONFIG_BATCH_SIZE, for_human=False)

            # print "before"
            # print wo_y_batch.shape
            # print wo_series_batch.shape
            # print user_x_batch.shape
            # print day_series_batch.shape

            ABC = None

            train_results = sess.run([
                alw.world_day_series_input,
                alw.world_workout_series_input,
                alw.world_y,
                alw.world_operation,
                alw.world_e,
                alw.world_workout_y,
                alw.world_combined,
                alw.world_combined_shaped,
                alw.world_b4shape,
                alw.world_lastA,
                alw.world_lastAA

                # alw.combined2,
                # alw.workout_y,
                # alw.afshape,
                # alw.user_vector_input
            ],

                feed_dict={
                    alw.world_day_series_input: day_series_batch,
                    alw.world_workout_series_input: wo_series_batch,
                    alw.world_user_vector_input: user_x_batch,
                    alw.world_workout_y: wo_y_batch
                })
            abc = None
            train_error += float(train_results[4])
            print "trainExtern: " + str(train_error / len(train_names))

        while len(valid_names_copy) > CONFIG.CONFIG_BATCH_SIZE:

            batch_unit_valid_names = valid_names_copy[:CONFIG.CONFIG_BATCH_SIZE]

            for ii in range(CONFIG.CONFIG_BATCH_SIZE):
                valid_names_copy.pop(0)

            day_series_batch = []
            user_x_batch = []
            workout_series_batch = []
            workout_y_batch = []

            wo_y_batch, wo_series_batch, user_x_batch, day_series_batch \
                = build_batch_from_names(batch_unit_valid_names, CONFIG.CONFIG_BATCH_SIZE, for_human=False)

            # print workout_y_batch.shape
            # print workout_series_batch.shape
            # print user_x_batch.shape
            # print day_series_batch.shape

            ABC = None

            valid_results = sess.run([
                alw.world_day_series_input,
                alw.world_workout_series_input,
                alw.world_y,
                # alw.world_operation,
                alw.world_e,
                alw.world_workout_y,
                alw.world_combined,
                alw.world_combined_shaped,
                alw.world_b4shape,
                alw.world_lastA,
                alw.world_lastAA

                # alw.combined2,
                # alw.workout_y,
                # alw.afshape,
                # alw.user_vector_input
            ],

                feed_dict={
                    alw.world_day_series_input: day_series_batch,
                    alw.world_workout_series_input: wo_series_batch,
                    alw.world_user_vector_input: user_x_batch,
                    alw.world_workout_y: wo_y_batch
                })
            abc = None
            valid_error += float(valid_results[3])
            # print "trainExtern: " + str(train_error)

        train_error /= float(len(train_names))
        valid_error /= float(len(valid_names))
        print "train_err: " + str(train_error) + " " + "valid_err: " + str(valid_error) + "best_err: " + str(best_error)

        # have to use this until you have enough samples
        # cuz atm 54 samples is not enough to generalize
        # and you need low low error

        if train_error < best_error:
            best_error = train_error
            print "model saved"
            alw.asaver.save(sess, CONFIG.CONFIG_SAVE_MODEL_LOCATION)

            # if train_error > valid_error:
            #    print "model saved"
            #    alw.asaver.save(sess,CONFIG.CONFIG_SAVE_MODEL_LOCATION)

    sess.close()


def train_rl_agent():
    all_names = get_unit_names()
    norm_vals = get_norm_values()

    loaded_unit = get_machine_unit_for_name(all_names[0], norm_vals)
    some_day_series = loaded_unit["dayseriesx"]
    some_user_x = loaded_unit["userx"]
    some_workout_series = loaded_unit["workoutxseries"]
    some_workout_y = loaded_unit["workouty"]

    RL_BATCH_SIZE = 1
    alw = Lift_NN(some_day_series, some_user_x, some_workout_series, some_workout_y, RL_BATCH_SIZE)

    init_op = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)

    # saver = tf.train.import_meta_graph(CONFIG.CONFIG_SAVE_MODEL_LOCATION+".meta")
    alw.asaver.restore(sess, tf.train.latest_checkpoint('/Users/admin/Desktop/tmp/'))

    # alw.asaver.restore(sess, CONFIG.CONFIG_SAVE_MODEL_LOCATION)


    # this shouldn't affect the stress model I think
    gradBuffer = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rl_agent'))

    # starting_point_name = []
    # starting_point_name.append(all_names[5])

    NUM_EPOCHS = 20000000

    reward_per_epoch = []

    for aepoch in range(NUM_EPOCHS):

        # so lets use each of these real datatpoints as a starting point
        # and let the model progress from there for a fixed number of steps?
        shuffle(all_names)

        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        reward_per_sample = []

        for a_sample_name in all_names:

            a_sample_name_batch = [a_sample_name]
            state = {}

            EPISODE_LENGTH = 40
            #EPISODE_LENGTH = 10

            actions_episode_log_human = []

            reward_episode = []
            action_index_episode = []
            value_episode = []
            dayseriesx_episode = []
            userx_episode = []
            workoutxseries_episode = []

            # then run the whole set for x epochs (add loop)
            # run an episode with each sample(add looop)
            for i in range(EPISODE_LENGTH):

                h_unit = {}

                if len(state.keys()) == 0:

                    wo_y_batch_h, wo_xseries_batch_h, user_x_batch_h, day_series_batch_h = build_batch_from_names(
                        a_sample_name_batch, 1, for_human=True)

                    # print wo_y_batch_h.shape
                    # print wo_xseries_batch_h.shape
                    # print user_x_batch_h.shape
                    # print day_series_batch_h.shape

                    h_unit["dayseriesx"] = day_series_batch_h[0]
                    h_unit["userx"] = user_x_batch_h[0]
                    h_unit["workoutxseries"] = wo_xseries_batch_h[0]

                    # we are bootstrapping with a training sample for training the rl
                    # so the last sample is hollowed out for the stress model to predict
                    # but here we want to make a decision with full data
                    # so we just throw off that last partial sample


                    npworkoutxseries = h_unit["workoutxseries"]
                    npworkoutxseries = np.delete(npworkoutxseries, len(npworkoutxseries) - 1)
                    h_unit["workoutxseries"] = npworkoutxseries

                    # h_unit["workouty"] = wo_y_batch_h[0]
                    h_unit["workouty"] = {}

                    state = {}
                    state["dayseriesx"] = h_unit["dayseriesx"]
                    state["userx"] = h_unit["userx"]
                    state["workoutxseries"] = h_unit["workoutxseries"]




                    state["lastrewarddetectedindexes"] = {}
                    for exercise_name in CHOOSABLE_EXERCISES:

                        # do not start last reward index where the rl starts taking actions
                        # cuz then it can just do 45lbsxreps then 1rmxreps for each episode
                        # to get reward
                        # we want it to take actions to increase a users max force over time
                        # so it should ideally look at the bootstrap sample past
                        # and try to improve upon the max reward seen in that past

                        # but what I am going to do here is just use the last seen
                        # exercise sample for now
                        # and just assume they are working up to a max or that the
                        # backoff sets are close enough
                        # or good enough of a jump off point that it nudges the RL
                        # to improve the user's max force instead of just doing
                        # repset patterns to get reward but not actually increase
                        # the user's max force over time

                        ex_index = exercise_vocabulary.index(exercise_name)
                        ex_key = "category_exercise_name_"+str(ex_index)

                        last_seen_index = None

                        for ik in range(len(h_unit["workoutxseries"])):
                            a_workout_step = h_unit["workoutxseries"][ik]
                            if a_workout_step[ex_key] == 1:
                                last_seen_index = ik

                        state["lastrewarddetectedindexes"][exercise_name] = last_seen_index




                    # need this one bc we want the rl to move from exercise to exercise
                    # so we implement a  penalty for when it keeps switching between exercises
                    # but for that we need to know when it started picking actions to take
                    state["rl_actions_started_index"] = len(h_unit["workoutxseries"])-1

                else:
                    h_unit["dayseriesx"] = state["dayseriesx"]
                    h_unit["userx"] = state["userx"]
                    h_unit["workoutxseries"] = state["workoutxseries"]
                    h_unit["workouty"] = {}
                    h_unit["lastrewarddetectedindexes"] = state["lastrewarddetectedindexes"]

                    h_unit["rl_actions_started_index"] = state["rl_actions_started_index"]-1
                    if h_unit["rl_actions_started_index"] < 0:
                        h_unit["rl_actions_started_index"] = 0


                m_unit = convert_human_unit_to_machine(h_unit, norm_vals)

                day_series_batch_m = [m_unit["dayseriesx"]]
                user_x_batch_m = [m_unit["userx"]]
                wo_xseries_batch_m = [m_unit["workoutxseries"]]
                wo_y_batch_m = [m_unit["workouty"]]

                ABC = None

                results = sess.run([
                    alw.agent_day_series_input,
                    alw.agent_workout_series_input,
                    alw.agent_policy1,
                    #alw.agent_value1,
                    #alw.agent_afshape,
                    #alw.agent_combined2,
                    alw.agent_user_vector_input,
                    #alw.copy_new_to_old,

                ],
                    feed_dict={
                        alw.agent_day_series_input: day_series_batch_m,
                        alw.agent_workout_series_input: wo_xseries_batch_m,
                        alw.agent_user_vector_input: user_x_batch_m
                    })

                # 0 bc we only run batch sizes of 1
                # so we can assume
                agent_softmax_choices = results[2][0]
                agent_value = results[3][0][0]

                # now just use the index of the highest softmax value to lookup the action
                # rl_all_possible_actions


                oai_index = np.random.choice(range(len(agent_softmax_choices)), p=agent_softmax_choices)
                oai_human_readable_action = rl_all_possible_actions[oai_index]

                # i think none of the probabilities ^^ are allowed to be zero
                # thats y u get the error then it collapses
                # need to do a check for that


                # oai_index = np.argmax(agent_softmax_choices)
                # oai_human_readable_action = rl_all_possible_actions[oai_index]

                rai_human_readable_action = np.random.choice(rl_all_possible_actions)
                rai_index = rl_all_possible_actions.index(rai_human_readable_action)

                human_readable_action = None
                action_index = None

                percent_done = 1.0  # float(aepoch)/float(NUM_EPOCHS)
                random_prob = 1.0 - percent_done
                not_random_prob = 1.0 - random_prob
                do_random_action = np.random.choice([True, False], p=[random_prob, not_random_prob])
                # do_random_action = np.random.choice([True, False], p=a_dist)


                # do_random_action = False
                # oai_index = np.random.choice(range(len(agent_softmax_choices)), p=agent_softmax_choices)
                # oai_human_readable_action = rl_all_possible_actions[oai_index]

                if do_random_action:
                    human_readable_action = rai_human_readable_action
                    action_index = rai_index
                else:
                    human_readable_action = oai_human_readable_action
                    action_index = oai_index

                # print human_readable_action


                # now pass the chosen action + state to the env
                action = human_readable_action
                state, reward, actions_episode_log_human = agent_world_take_step(state, action, alw, sess,actions_episode_log_human)

                value_episode.append(agent_value)
                reward_episode.append(reward)
                action_index_episode.append(action_index)
                dayseriesx_episode.append(m_unit["dayseriesx"][:])
                userx_episode.append(m_unit["userx"][:])
                workoutxseries_episode.append(m_unit["workoutxseries"][:])

            reward_per_sample.append(np.sum(reward_episode))

            gamma = 0.99

            def discount_rewards(r):
                # the strength with which we encourage a sampled action is the weighted sum of all
                # rewards afterwards
                """ take 1D float array of rewards and compute discounted reward """
                discounted_r = np.zeros_like(r)
                running_add = 0
                for t in reversed(xrange(0, r.size)):
                    running_add = running_add * gamma + r[t]
                    discounted_r[t] = running_add
                return discounted_r

            # https://arxiv.org/pdf/1506.02438.pdf
            # advantage function page 2
            # A(s,a) = Q(s,a)-V(s)
            # measures whether or not the action is better or worse than the policys default behavior
            # A(s,a) is not known and has to be estimated
            # page 4 you can use an advantage estimator
            # Aestimate = r[t] + (gamma * v[t+1]) - v[t]
            # below we are basically doing
            # number 18 from page 5
            # doing number 18 bc it has lower variance

            advantages_episode = []
            advantages_episode = np.array(advantages_episode)
            for i in range(len(reward_episode) - 1):
                adv = reward_episode[i] + (gamma * value_episode[i + 1]) - value_episode[i]
                advantages_episode = np.append(advantages_episode, [adv])
            advantages_episode = np.append(advantages_episode, [0])

            advantages_episode = discount_rewards(advantages_episode)
            dreward_episode = discount_rewards(np.array(reward_episode))

            preward = np.array(dreward_episode)
            paction = np.array(action_index_episode)
            pvalue = np.array(value_episode)
            padvantages = np.array(advantages_episode)
            pusersx = np.array(userx_episode)
            pworkoutxseries = np.array(workoutxseries_episode)
            pdayseriesx = np.array(dayseriesx_episode)

            feed_dict = {
                alw.reward_holder: preward,
                alw.action_holder: paction,
                alw.value_holder: pvalue,
                alw.advantage_holder: padvantages
                ,
                alw.agent_day_series_input: pdayseriesx,
                alw.agent_workout_series_input: pworkoutxseries,
                alw.agent_user_vector_input: pusersx
            }

            results1 = sess.run([
                alw.gradient,
                alw.reward_holder,
                alw.action_holder,
                alw.value_holder,
                alw.policy_loss1,
                #alw.value_loss,
                alw.loss
                #,
                #alw.ratio
            ], feed_dict=feed_dict)


            #print results1[4]
            #print results1[5]
            #print results1[6]

            grads = results1[0]
            #print str(results1[4])+" "+str(results1[5])

            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            #print "env sample_over-----------------------------------------------------------"
            #print out human actions here if you want
                #actions_episode_log_human


        results2 = sess.run([alw.copy_new_to_old], feed_dict=feed_dict)

        feed_dict = dict(zip(alw.gradient_holders, gradBuffer))
        results1 = sess.run([alw.update_batch], feed_dict=feed_dict)



        rps = np.mean(reward_per_sample)
        reward_per_epoch.append(rps)
        print str(aepoch) + " " + str(rps) + " " + str(np.mean(reward_per_epoch))

    print reward_per_epoch


# exercise=squat:reps=6:weight=620

def agent_world_take_step(state, action, ai_graph, sess,actions_episode_log_human):
    # run through the lift world
    # make a new state
    # check for reward

    # need to parse action and insert it into the liftworld input
    # get the output

    a_day_series_h = state['dayseriesx']
    a_user_x_h = state['userx']
    a_workout_series_h = state['workoutxseries']

    # ----------------------------------------------------------

    # make a workout vector unit from chosen action

    action_exercise_name_human = (action.split(":")[0]).split("=")[1]
    action_planned_reps_human = (action.split(":")[1]).split("=")[1]
    action_multiplier_lbs_human = (action.split(":")[2]).split("=")[1]

    state_h = state
    reward = 0

    if action_exercise_name_human == "LEAVEGYM":
        # can just take the last day and add it again

        a_day_series_h = np.append(a_day_series_h, copy.deepcopy(a_day_series_h[-1]))
        a_day_series_h = np.delete(a_day_series_h, 0)

        # a_day_series_h.append(a_day_series_h[-1][:])

        state_h['dayseriesx'] = a_day_series_h
        state_h['userx'] = state['userx']
        state_h['workoutxseries'] = state['workoutxseries']
        state_h["rl_actions_started_index"] = len(state['workoutxseries'])-1

        actions_episode_log_human.append("env LEAVEGYM")
        #print "env LEAVEGYM"

    if action_exercise_name_human != "LEAVEGYM":
        # we will let nn calc rest intervals from
        # times from the start of workout

        exercise_name = action_exercise_name_human
        reps_planned = action_planned_reps_human
        reps_completed = -1

        last_weight_lbs = float(a_workout_series_h[-1]["weight_lbs"])
        new_weight_lbs = last_weight_lbs + (float(action_multiplier_lbs_human)*last_weight_lbs)

        MINIMUM_WEIGHT = 45.0
        MAXIMUM_WEIGHT = 1000.0
        if new_weight_lbs < MINIMUM_WEIGHT:
            new_weight_lbs = MINIMUM_WEIGHT
        if new_weight_lbs > MAXIMUM_WEIGHT:
            new_weight_lbs = MAXIMUM_WEIGHT

        weight_lbs = new_weight_lbs

        #print "env "+exercise_name+" "+str(new_weight_lbs)+" "+str(reps_planned)
        actions_episode_log_human.append("env "+exercise_name+" "+str(new_weight_lbs)+" "+str(reps_planned))


        postset_heartrate = -1
        went_to_failure = -1
        did_pull_muscle = -1

        used_lifting_gear = -1
        unit_days_arr_human = a_day_series_h
        velocities_m_per_s_arr = -1

        workoutstep_for_predict_h = make_workout_step_human(
            action_exercise_name_human,
            action_planned_reps_human,
            reps_completed,
            weight_lbs,

            postset_heartrate,
            went_to_failure,
            did_pull_muscle,

            used_lifting_gear,
            unit_days_arr_human,
            velocities_m_per_s_arr
        )

        # np array so u cant use append like above
        a_workout_series_h = np.append(a_workout_series_h, workoutstep_for_predict_h)

        # ----------------------------------------------------------

        # still need to pad and trim to max lengths
        # ez to do, just remove 1 from the back if you are appending 1
        # we dont have to do this (bc its only one item in the batch, so it can
        # have any shape )
        #  to pass it through the network
        # but the network was trained with a certain length
        # so we should structure the data simliar to how it was trained

        a_workout_series_h = np.delete(a_workout_series_h, 0)

        # ----------------------------------------------------------
        # convert state to machine format

        state_h = {}
        state_h['dayseriesx'] = a_day_series_h
        state_h['userx'] = a_user_x_h
        state_h['workoutxseries'] = a_workout_series_h
        state_h['workouty'] = {}

        norm_vals = get_norm_values()
        state_m = convert_human_unit_to_machine(state_h, norm_vals)

        day_series_batch = [state_m["dayseriesx"]]
        user_x_batch = [state_m["userx"]]
        workout_series_batch = [state_m["workoutxseries"]]

        day_series_batch = np.array(day_series_batch)
        user_x_batch = np.array(user_x_batch)
        workout_series_batch = np.array(workout_series_batch)

        # ---------------------------------------------------------
        # put through graph
        alw = ai_graph

        # print workout_y_batch.shape
        # print workout_series_batch.shape
        # print user_x_batch.shape
        # print day_series_batch.shape

        results_of_action = sess.run([
            alw.world_day_series_input,
            alw.world_workout_series_input,
            alw.world_y
        ],

            feed_dict={
                alw.world_day_series_input: day_series_batch,
                alw.world_workout_series_input: workout_series_batch,
                alw.world_user_vector_input: user_x_batch
            })

        m_filled_workout_step = results_of_action[2][0]

        h_filled_workout_step = make_h_workout_with_xh_ym(workoutstep_for_predict_h, m_filled_workout_step,
                                                          a_day_series_h)

        # remove the partial one we added to the end so we could predict
        a_workout_series_h = np.delete(a_workout_series_h, len(a_workout_series_h) - 1)

        # append the filled one to the end
        a_workout_series_h = np.append(a_workout_series_h, h_filled_workout_step)

        # make the new state with the filled result of the action
        state_h = {}
        state_h['dayseriesx'] = a_day_series_h
        state_h['userx'] = a_user_x_h
        state_h['workoutxseries'] = a_workout_series_h
        state_h['workouty'] = {}
        state_h["lastrewarddetectedindexes"] = state["lastrewarddetectedindexes"]
        state_h["rl_actions_started_index"] = state["rl_actions_started_index"]

        # print a_workout_series_h[-1]

        # now calculate reward from the last reward index----------------


        # find max force for index
        # compare to max force at latest index
        # see if max force increases
        # if it does, save the new lastrewarddetectedindex

        # i think you should use the average velocity
        # of completed reps instead of individual resps
        # to calculate whether force increased
        # this is because there is variance in the sensor measurements
        # for deadlift, I think sometimes the rubber weights make it
        # bounce a little and the bar speed sensor records that

        # I am calculating continuous reward instead of end of episode reward
        # bc I think it helps the rl agent learn exactly what is good
        # discounted return doesnt provide the rl agent much info on what
        # it did well if you just calculate the reward at the end



        # we store a last reward calculated index for each exercise
        # so when calculating reward per exercise the reward is only calculated
        # from that exercise
        # aka dont see a force increase from benchpress when you do deadlift and count
        # it as reward



        rl_exercise_chosen_h = action_exercise_name_human
        no_reward_just_set_last_reward_detected_index = False
        start_indexa = state_h["lastrewarddetectedindexes"][rl_exercise_chosen_h]

        if start_indexa is None:
            no_reward_just_set_last_reward_detected_index = True

        start_workout_force = None
        latest_workout_force = None

        # have to do this bc the timeseries length is a fixed moving window
        # so when we move the window we need to decrement the lastrewardindex
        # for each exercise
        # bc by the time we have reached here we have moved the window
        keys = state_h["lastrewarddetectedindexes"].keys()
        for akey in keys:
            checkNone = state_h["lastrewarddetectedindexes"][akey]
            if checkNone is not None:
                state_h["lastrewarddetectedindexes"][akey] = state_h["lastrewarddetectedindexes"][akey] - 1
                if state_h["lastrewarddetectedindexes"][akey] < 0:
                    state_h["lastrewarddetectedindexes"][akey] = None

        #the above can set a exercise to none
        start_indexb = state_h["lastrewarddetectedindexes"][rl_exercise_chosen_h]
        if start_indexb is None:
            no_reward_just_set_last_reward_detected_index = True


        if no_reward_just_set_last_reward_detected_index == False:

            start_indexc = state_h["lastrewarddetectedindexes"][rl_exercise_chosen_h]

            # if start_index

            start_workout_step = state_h["workoutxseries"][start_indexc]
            start_workout_reps_completed = start_workout_step["reps_completed"]
            start_workout_weight_lbs = float(start_workout_step["weight_lbs"])
            start_workout_velocities = []
            for iiii in range(int(math.floor(start_workout_reps_completed))):
                a_velocity = float(start_workout_step["velocities_arr_" + str(iiii)])
                start_workout_velocities.append(a_velocity)

            # when we make the train units that we bootstrap this with
            # we pad the units
            # so here when it is first bootstrapped it can see those padded values here
            # they will show up as nan
            # if that happens here I suppose we just set the index to calculate the next
            # reward from

            start_workout_force = None
            no_reward_just_set_last_reward_detected_index = False
            if len(start_workout_velocities) == 0:
                no_reward_just_set_last_reward_detected_index = True
            else:
                start_workout_average_velocity = np.mean(start_workout_velocities)
                start_workout_force = float(start_workout_average_velocity) * float(start_workout_weight_lbs)

            latest_workout_step = state_h["workoutxseries"][-1]
            latest_workout_reps_completed = latest_workout_step["reps_completed"]
            latest_workout_weight_lbs = float(latest_workout_step["weight_lbs"])
            latest_workout_velocities = []
            for iiii in range(int(math.floor(latest_workout_reps_completed))):
                a_velocity = latest_workout_step["velocities_arr_" + str(iiii)]
                latest_workout_velocities.append(a_velocity)
            latest_workout_average_velocity = np.mean(latest_workout_velocities)
            latest_workout_force = latest_workout_average_velocity * latest_workout_weight_lbs

            # force here units are in lbs per meters/second oh boy
            # should convert fully to metric but don't really need to
            # bc rl agent doesnt really care about what units
            # reward is in

        reward = 0
        if no_reward_just_set_last_reward_detected_index:
            state_h["lastrewarddetectedindexes"][rl_exercise_chosen_h] = len(state_h["workoutxseries"]) - 1
        else:
            new_reward = latest_workout_force - start_workout_force

            if new_reward > 0:
                reward = new_reward
                state_h["lastrewarddetectedindexes"][rl_exercise_chosen_h] = len(state_h["workoutxseries"]) - 1

                # print str(latest_workout_force)+" "+str(start_workout_force)+" "+str(len(state_h["workoutxseries"])-1)+" "+\
                #      str(state_h["lastrewarddetectedindex"])
                # print rl_exercise_chosen_h + " " + str(new_reward)+" "+ str(len(state_h["workoutxseries"])-1) +" : "+str(start_index)
                # print new_reward

        #------------------------------------------------------------------------------------------------------------
        # test to make sure the RL agent doesnt mix exercise
        # we want it to move from one to the next
        # so we put in a penalty if it keeps switching
        # so you count one continuous exercise range as a block
        # for every exercise range over 3 you minus some reward
        # well, you can just count how many times it changes
        # that might be simpler
        # just scan forwards and each time it changes you increment a counter

        continuous_exercises_check_arr = state_h["workoutxseries"][state_h["rl_actions_started_index"]:len(state_h["workoutxseries"])-1]

        num_times_exercise_changed = 0
        last_workoutstep_exercise_key_index = None

        for i in range(len(continuous_exercises_check_arr)):
            workoutstep = continuous_exercises_check_arr[i]
            workoutstep_exercise_keys = []
            for key in workoutstep.keys():
                if "category_exercise_name_" in key:
                    workoutstep_exercise_keys.append(key)
            are_any_exercises_picked = False
            for key in workoutstep_exercise_keys:
                if workoutstep[key] == 1:
                    key_index = workoutstep.index(key)
                    if last_workoutstep_exercise_key_index is None:
                        last_workoutstep_exercise_key_index = key_index
                    else:
                        if last_workoutstep_exercise_key_index != key_index:
                            num_times_exercise_changed = num_times_exercise_changed + 1

        # using a dumb method
        # really we should check to see all the exercises chosen so far
        # and then do it based off of that
        # this also penalizes for the user doing exercises that the rl agent cant choose
        # it gives reward if rl doesnt do all of the exercises, eg it gives reward of 2 if
        # the rl picks benchpress and skips dead+squat
        mixed_reward_penalty = len(CHOOSABLE_EXERCISES) - num_times_exercise_changed

        reward = reward + (mixed_reward_penalty*10)

        #------------------------------------------------------------------------------------------------------------

        # put in a penalty for each set
        # so the NN encourages the user to leave the gym when
        # it knows doing extra sets will not give a reward

        excessive_sets_penalty = -10
        reward = reward + excessive_sets_penalty

        #------------------------------------------------------------------------------------------------------------

        # penalize if completed reps is less than planned reps
        # so penalize it by the average speed of completed reps * missed reps

        last_workout_step = state_h["workoutxseries"][-1]
        last_completed_reps = last_workout_step["reps_completed"]
        last_planned_reps = last_workout_step["reps_planned"]
        last_weight = float(last_workout_step["weight_lbs"])
        missed_reps = (float(last_planned_reps) - float(last_completed_reps))
        last_workout_velocities = []
        for iiii in range(int(math.floor(last_completed_reps))):
            a_velocity = float(last_workout_step["velocities_arr_"+str(iiii)])
            last_workout_velocities.append(a_velocity)

        last_avg_velocity = np.mean(last_workout_velocities)

        missed_reps_penalty = last_avg_velocity * missed_reps

        reward = reward - missed_reps_penalty




        #------------------------------------------------------------------------------------------------------------

        ABC = None

    return state_h, reward, actions_episode_log_human


#train_stress_adaptation_model()
train_rl_agent()









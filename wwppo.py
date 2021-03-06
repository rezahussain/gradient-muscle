
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
# https://www.linkedin.com/in/reza-hussain-34430066
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
import warnings as warnings
import sys

import CONFIG as CONFIG


def get_dir_names(dir):
    return filter(lambda f: not f.startswith('.'), os.listdir(dir))

class UserData():
    def __init__(self,a_user_name):

        self.user_name = a_user_name

        self.json_raw_paths = []
        json_filenames = get_dir_names(CONFIG.CONFIG_RAW_JSON_PATH + a_user_name)
        for a_json_filename in json_filenames:
            ajsonpath = CONFIG.CONFIG_RAW_JSON_PATH + a_user_name + "/" + a_json_filename
            self.json_raw_paths.append(ajsonpath)
        self.json_raw_paths.sort()

        self.user_jos = []
        for i in range(len(self.json_raw_paths)):
            d = open(self.json_raw_paths[i])
            o = json.load(d)
            self.user_jos.append(o)

        # we build one datapoint for each workoutday
        # it consists of
        # the workout
        # then the last workout
        # then all of the day vectors on the side in the range
        # so we build a range for each datapoint
        # based off of the workout lookback

        self.workout_indexes = []
        for i in range(len(self.user_jos)):
            jo = self.user_jos[i]
            if len(jo['workout_vector_arr']) > 0:
                self.workout_indexes.append(i)

        self.workout_ranges = []
        for i in range(len(self.workout_indexes)):
            behind = self.workout_indexes[:i]
            if len(behind) > CONFIG.CONFIG_WORKOUT_LOOKBACK:
                end = self.workout_indexes[i]
                start = self.workout_indexes[i - CONFIG.CONFIG_WORKOUT_LOOKBACK]
                self.workout_ranges.append([start, end])




user_folders = get_dir_names(CONFIG.CONFIG_RAW_JSON_PATH)

global_user_objs = []
for user_folder in user_folders:
    new_user = UserData(user_folder)
    global_user_objs.append(new_user)

global_pulled_muscle_vocabulary = []
global_max_workout_range_array_len = 0
global_exercise_vocabulary = []
global_max_day_range_array_len = 0

for a_user_obj in global_user_objs:

    # ---------------------------------------------------------->
    # build a vocabulary for the pulled muscle name

    u_workout_indexes = a_user_obj.workout_indexes
    u_json_objects = a_user_obj.user_jos
    u_workout_ranges = a_user_obj.workout_ranges

    for i in u_workout_indexes:
        jo = u_json_objects[i]
        workout_vector = jo["workout_vector_arr"]
        for set in workout_vector:
            if set["pulled_muscle_name"] not in global_pulled_muscle_vocabulary:
                global_pulled_muscle_vocabulary.append(set["pulled_muscle_name"])

    # ---------------------------------------------------------->
    # need to build exercise name vocabulary

    for i in u_workout_indexes:
        jo = u_json_objects[i]
        workout_vector = jo["workout_vector_arr"]
        for set in workout_vector:
            if set["exercise_name"] not in global_exercise_vocabulary:
                global_exercise_vocabulary.append(set["exercise_name"])

    # ---------------------------------------------------------->

    # need to find max workout_arr length so we can pad
    # all of the workouts to this length
    # so we can train in batches
    # bc batches need the same shapes

    for r in u_workout_ranges:
        workout_range_max_num_sets = 0
        for xx in range(r[0], r[1] + 1):
            workout_range_max_num_sets += len(u_json_objects[xx]["workout_vector_arr"])
        if workout_range_max_num_sets > global_max_workout_range_array_len:
            global_max_workout_range_array_len = workout_range_max_num_sets

    Abc = None

    # ---------------------------------------------------------->

    # now we need to find max range day length so we can pad
    # all of the day series units to this length
    # again so we can train in batches
    # bc batches need the same shape

    for r in u_workout_ranges:
        #rdays = r[1] - r[0]
        daysforrange = u_json_objects[r[0]:r[1]+1]#add 1 so the slice includes the last item
        rdays = len(daysforrange)
        if rdays > global_max_day_range_array_len:
            global_max_day_range_array_len = rdays

    ABC = None

    #so because you can have ranges like 0,10
    #it actually has 11 elements bc u start at 0
    #so 10-0 gives 10 which is wrong
    #then later 11 shows up which is correct
    #and messes up the day_series batch size bc make raw units
    # it pads to 10 instead of 11
    #and there are 11 size elements in the mix
    #so here we add 1
    #to get the right num of elements when range is something like 0,10 (11 units bc u count 0)
    #you can also get the elements using the range and then do a len
    #but im just doing this here
    #global_max_day_range_array_len += 1
    #i actually decided to get the elements using the range then do a len

    # ---------------------------------------------------------->




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


# weight is in lbs
CHOOSABLE_EXERCISES = ["squat", "benchpress", "deadlift"]

REP_ADJUSTMENT_ACTIONS = [0,1,2,3,4,-1,-2,-3,-4]
WEIGHT_ADJUSTMENT_ACTIONS = [0,5,10,50,70,90,-5,-10,-50,-70,-90]
# now we need to make all of the combos the RLAgent can pick

rl_all_possible_actions = []

ADJUST_REPS = "ADJUST_REPS"
ADJUST_WEIGHT = "ADJUST_WEIGHT"

for y in REP_ADJUSTMENT_ACTIONS:
    rl_all_possible_actions.append(ADJUST_REPS+"="+str(y))
for y in WEIGHT_ADJUSTMENT_ACTIONS:
    rl_all_possible_actions.append(ADJUST_WEIGHT+"="+str(y))

LEAVE_GYM = "LEAVEGYM"
NEXT_EXERCISE = "NEXTEXERCISE"
rl_all_possible_actions.append(LEAVE_GYM)
rl_all_possible_actions.append(NEXT_EXERCISE)

# RL has a hard time learning
# when there are many choices

#---------------------------------------------------------->

def convert_raw_json_day_vector_to_packaged_day(index,
                                                some_json_objects,
                                                last_day_vector_workout_day_index,
                                                unit_days,
                                                previous_day_index):

    # build day array
    # print xx
    xx = index
    packaged_day = {}
    packaged_day["heart_rate_variability_rmssd"] = some_json_objects[xx]["day_vector"]["heart_rate_variability_rmssd"]
    packaged_day["post_day_wearable_calories_burned"] = some_json_objects[xx]["day_vector"][
        "post_day_wearable_calories_burned"]
    packaged_day["post_day_calories_in"] = some_json_objects[xx]["day_vector"]["post_day_calories_in"]
    packaged_day["post_day_protein_g"] = some_json_objects[xx]["day_vector"]["post_day_protein_g"]
    packaged_day["post_day_carbs_g"] = some_json_objects[xx]["day_vector"]["post_day_carbs_g"]
    packaged_day["post_day_fat_g"] = some_json_objects[xx]["day_vector"]["post_day_fat_g"]
    packaged_day["withings_weight_lbs"] = some_json_objects[xx]["day_vector"]["withings_weight_lbs"]
    packaged_day["withings_body_fat_percent"] = some_json_objects[xx]["day_vector"]["withings_body_fat_percent"]
    packaged_day["withings_muscle_mass_percent"] = some_json_objects[xx]["day_vector"]["withings_muscle_mass_percent"]
    packaged_day["withings_body_water_percent"] = some_json_objects[xx]["day_vector"]["withings_body_water_percent"]
    packaged_day["withings_heart_rate_bpm"] = some_json_objects[xx]["day_vector"]["withings_heart_rate_bpm"]
    packaged_day["withings_bone_mass_percent"] = some_json_objects[xx]["day_vector"]["withings_bone_mass_percent"]
    packaged_day["withings_pulse_wave_velocity_m_per_s"] = some_json_objects[xx]["day_vector"][
        "withings_pulse_wave_velocity_m_per_s"]

    sbtap = time_vocabulary.index(some_json_objects[xx]["day_vector"]["sleeptime_bed_time_ampm"])
    packaged_day["sleeptime_bed_time_ampm_index"] = sbtap

    srtap = time_vocabulary.index(some_json_objects[xx]["day_vector"]["sleeptime_rise_time_ampm"])
    packaged_day["sleeptime_rise_time_ampm_index"] = srtap

    packaged_day["sleeptime_efficiency_percent"] = some_json_objects[xx]["day_vector"]["sleeptime_efficiency_percent"]

    sarap = time_vocabulary.index(some_json_objects[xx]["day_vector"]["sleeptime_alarm_ring_ampm"])
    packaged_day["sleeptime_alarm_ring_ampm_index"] = sarap

    sasap = time_vocabulary.index(some_json_objects[xx]["day_vector"]["sleeptime_alarm_set_ampm"])
    packaged_day["sleeptime_alarm_set_ampm_index"] = sasap

    packaged_day["sleeptime_snoozed"] = some_json_objects[xx]["day_vector"]["sleeptime_snoozed"]

    def getNumericalHour(hour_string):
        result = None
        if hour_string != -1:
            hours = hour_string.split(":")[0]
            minutes = hour_string.split(":")[1]
            result = float(hours) + (float(minutes) / 60.0)
        else:
            result = -1
        return result

    sahrs = getNumericalHour(some_json_objects[xx]["day_vector"]["sleeptime_awake_hrs"])
    packaged_day["sleeptime_awake_hrs"] = sahrs

    slshrs = getNumericalHour(some_json_objects[xx]["day_vector"]["sleeptime_light_sleep_hrs"])
    packaged_day["sleeptime_light_sleep_hrs"] = slshrs

    sdrhrs = getNumericalHour(some_json_objects[xx]["day_vector"]["sleeptime_deep_rem_hrs"])
    packaged_day["sleeptime_deep_rem_hrs"] = sdrhrs

    if last_day_vector_workout_day_index is not None:
        current_workout_yyyymmdd = some_json_objects[xx]["day_vector"]["date_yyyymmdd"]
        last_workout_yyyymmdd = some_json_objects[last_day_vector_workout_day_index]["day_vector"]["date_yyyymmdd"]
        days_since_last_workout = calc_days_between_dates(current_workout_yyyymmdd, last_workout_yyyymmdd)
    else:
        days_since_last_workout = 0
    packaged_day["days_since_last_workout"] = days_since_last_workout
    if len(some_json_objects[xx]["workout_vector_arr"]) > 0:
        last_day_vector_workout_day_index = xx


    current_day_yyyymmdd = some_json_objects[xx]["day_vector"]["date_yyyymmdd"]
    last_recorded_day_yyyymmdd = some_json_objects[previous_day_index]["day_vector"]["date_yyyymmdd"]
    days_since_last_recorded_day = calc_days_between_dates(current_day_yyyymmdd, last_recorded_day_yyyymmdd)

    packaged_day["days_since_last_recorded_day"] = days_since_last_recorded_day

    packaged_day["day_number"]=len(unit_days)

    return packaged_day,last_day_vector_workout_day_index


def make_raw_units():


    pulled_muscle_dates = []
    waiting_on_pulled_muscle = True


    for u in global_user_objs:

        some_workout_ranges = u.workout_ranges

        some_json_filenames = u.json_raw_paths

        some_json_objects = u.user_jos

        for r in some_workout_ranges:

            # make a unit for each range
            unit = {}
            unit_user = {}
            unit_days = []
            unit_workouts = []
            unit_y = []

            per_set_workout = []

            last_day_vector_workout_day_index = None

            body_train_unit = None




            for xx in range(r[0], r[1] + 1):

                debug_name = some_json_filenames[xx]

                previous_day_index = None
                if xx == r[0]:
                    previous_day_index = xx
                if xx > r[0]:
                    previous_day_index = xx-1

                packaged_day,last_day_vector_workout_day_index = \
                    convert_raw_json_day_vector_to_packaged_day(xx,
                                                                some_json_objects,
                                                                last_day_vector_workout_day_index,
                                                                unit_days,
                                                                previous_day_index)




                # if you modify the setup above make sure it is also modded
                # in the pad unit_days below
                unit_days.append(packaged_day)

                # need to do it as one big one
                # bc its easier to make sure that you do not include
                # days that are ahead of the current work set
                # but then still include days that are behind or at the current
                # work set

                # save the last work set day whole unit
                # bc its all of the sets for the day that are responsible for
                # the body state the next day






                if len(some_json_objects[xx]["workout_vector_arr"]) > 0:

                    for ii in range(len(some_json_objects[xx]["workout_vector_arr"])):

                        exercise_name = some_json_objects[xx]["workout_vector_arr"][ii]["exercise_name"]
                        reps_planned = some_json_objects[xx]["workout_vector_arr"][ii]["reps_planned"]
                        reps_completed = some_json_objects[xx]["workout_vector_arr"][ii]["reps_completed"]
                        weight_lbs = some_json_objects[xx]["workout_vector_arr"][ii]["weight_lbs"]

                        postset_heartrate = some_json_objects[xx]["workout_vector_arr"][ii]["postset_heartrate"]
                        went_to_failure = some_json_objects[xx]["workout_vector_arr"][ii]["went_to_failure"]
                        did_pull_muscle = some_json_objects[xx]["workout_vector_arr"][ii]["did_pull_muscle"]

                        used_lifting_gear = some_json_objects[xx]["workout_vector_arr"][ii]["used_lifting_gear"]
                        vmpsa = some_json_objects[xx]["workout_vector_arr"][ii]["velocities_m_per_s_arr"]


                        if did_pull_muscle and waiting_on_pulled_muscle:
                            p_day_yyyymmdd = some_json_objects[xx]["day_vector"]["date_yyyymmdd"]
                            if p_day_yyyymmdd not in pulled_muscle_dates:
                                pulled_muscle_dates.append(p_day_yyyymmdd)
                                waiting_on_pulled_muscle = False
                        if not waiting_on_pulled_muscle:
                            p_day_yyyymmdd = some_json_objects[xx]["day_vector"]["date_yyyymmdd"]
                            if p_day_yyyymmdd not in pulled_muscle_dates:
                                pulled_muscle_dates.append(p_day_yyyymmdd)
                                waiting_on_pulled_muscle = True


                        # lets say you have days_since_last_workout all 1
                        # it cant know when a new workout started
                        # it probably thinks of it all as one workout
                        # so we add the set number of the day
                        # instead of using time since start of workout
                        # bc if we use time since start of workout we have to do
                        # things like estimate how long a user is resting

                        # this whole spiel is a consequence of using a separate
                        # timeseries vector for days and workouts instead of combining
                        # them
                        # we could combine them, but then we have to predict both of the
                        # missing pieces at the same time and that gets messy
                        # because of things like doing a set it might try to predict
                        # the day values
                        # I think keeping them separate timeseries is the right approach
                        # at least thats how it appears to me right now

                        set_number_of_the_day = ii

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
                            vmpsa,
                            set_number_of_the_day
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
                        for iii in range(len(global_exercise_vocabulary)):
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

                        while len(unit_workout_clone_padded) < global_max_workout_range_array_len:
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
                        packaged_day_padded["day_number"] = -1
                        packaged_day_padded["days_since_last_recorded_day"] = -1

                        while len(unit_days_padded) < global_max_day_range_array_len:
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

                        userjson = some_json_objects[xx]["user_vector"]
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

                        whole_train_unit = {}
                        whole_train_unit["workoutxseries"] = workoutxseries
                        whole_train_unit["workouty"] = workouty
                        whole_train_unit["dayseriesx"] = dayseries
                        whole_train_unit["userx"] = userx

                        if has_valid_y:
                            savename = some_json_objects[xx]["day_vector"]["date_yyyymmdd"] + "_" + str(ii)
                            savename = savename + u.user_name
                            pickle.dump(whole_train_unit, open(CONFIG.CONFIG_NN_STRESS_MODEL_PICKLES_PATH + savename, "wb"))

                        body_train_unit = copy.deepcopy(whole_train_unit)


                if xx + 1 < len(some_json_objects) - 1 and body_train_unit is not None:

                    current_day_yyyymmdd = some_json_objects[xx+1]["day_vector"]["date_yyyymmdd"]
                    last_recorded_day_yyyymmdd = some_json_objects[xx]["day_vector"]["date_yyyymmdd"]
                    days_since_last_recorded_day = calc_days_between_dates(current_day_yyyymmdd,
                                                                           last_recorded_day_yyyymmdd)

                    # make body trainer only predict day after body state
                    # do not feed it data where the day it is predicting is
                    # like 3 days in the future
                    # we only want it to predict one day into the future
                    # since that is what we do when we do next_day in the rl
                    # agent so that is what we want to simulate

                    if days_since_last_recorded_day == 1:
                        packaged_day_after, last_day_vector_workout_day_index = \
                            convert_raw_json_day_vector_to_packaged_day(xx+1, some_json_objects,
                                            last_day_vector_workout_day_index, unit_days,xx)
                        packaged_dayy = {}
                        packaged_dayy["withings_weight_lbs"] = packaged_day_after["withings_weight_lbs"]
                        packaged_dayy["withings_muscle_mass_percent"] = packaged_day_after["withings_muscle_mass_percent"]
                        packaged_dayy["withings_body_fat_percent"] = packaged_day_after["withings_body_fat_percent"]
                        packaged_dayy["withings_body_water_percent"] = packaged_day_after["withings_body_water_percent"]
                        packaged_dayy["withings_heart_rate_bpm"] = packaged_day_after["withings_heart_rate_bpm"]
                        packaged_dayy["withings_bone_mass_percent"] = packaged_day_after["withings_bone_mass_percent"]
                        packaged_dayy["withings_pulse_wave_velocity_m_per_s"] = packaged_day_after["withings_pulse_wave_velocity_m_per_s"]

                        body_train_unit["dayy"] = packaged_dayy

                        has_all_body_y = False
                        if (
                            packaged_day["withings_weight_lbs"] is not -1
                            and packaged_day["withings_muscle_mass_percent"] is not -1
                            and packaged_day["withings_body_water_percent"] is not -1
                            and packaged_day["withings_heart_rate_bpm"] is not -1
                            and packaged_day["withings_bone_mass_percent"] is not -1
                            and packaged_day["withings_pulse_wave_velocity_m_per_s"] is not -1
                        ):
                            has_all_body_y = True


                        if has_all_body_y:
                            savename = some_json_objects[xx]["day_vector"]["date_yyyymmdd"] + "_"
                            savename = savename + u.user_name
                            pickle.dump(body_train_unit, open(CONFIG.CONFIG_NN_BODY_MODEL_PICKLES_PATH + savename, "wb"))



    #now figure out avg_pulled_muscle time
    days_between_pull_recovery = []
    pp = 0
    if len(pulled_muscle_dates) % 2 == 1:
        pulled_muscle_dates.pop()
    while pp < len(pulled_muscle_dates):
        pulled_date = pulled_muscle_dates[pp+0]
        resume_date = pulled_muscle_dates[pp+1]
        pprr = calc_days_between_dates(pulled_date,resume_date)
        days_between_pull_recovery.append(pprr)
        pp = pp+2

    days_between_pull_recovery = np.array(days_between_pull_recovery)
    num_days_between_pull_recovery = np.mean(days_between_pull_recovery)

    metadata = {}
    metadata["days_between_pull_and_recovery"] = num_days_between_pull_recovery
    pickle.dump(metadata, open(CONFIG.CONFIG_METADATA_PATH, "wb"))
    ABC = None



def get_num_days_between_pull_recovery():
    if get_num_days_between_pull_recovery.GLOBAL_DAYS_BETWEEN_PULL_RECOVERY is None:
        metadata = pickle.load(open(CONFIG.CONFIG_METADATA_PATH,"rb"))
        val_we_want = metadata["days_between_pull_and_recovery"]
        get_num_days_between_pull_recovery.GLOBAL_DAYS_BETWEEN_PULL_RECOVERY=val_we_want
    return get_num_days_between_pull_recovery.GLOBAL_DAYS_BETWEEN_PULL_RECOVERY
get_num_days_between_pull_recovery.GLOBAL_DAYS_BETWEEN_PULL_RECOVERY = None





# ---------------------------------------------------------->

def calc_days_between_dates(current_yyyymmdd, last_yyyymmdd):
    cwyyyy = int(current_yyyymmdd[0:4])
    cwmm = int(current_yyyymmdd[4:6])
    cwdd = int(current_yyyymmdd[6:8])

    lwyyyy = int(last_yyyymmdd[0:4])
    lwmm = int(last_yyyymmdd[4:6])
    lwdd = int(last_yyyymmdd[6:8])

    cw_datetime = datetime.datetime(cwyyyy, cwmm, cwdd)
    lw_datetime = datetime.datetime(lwyyyy, lwmm, lwdd)

    cw_timestamp = calendar.timegm(cw_datetime.timetuple())
    lw_timestamp = calendar.timegm(lw_datetime.timetuple())

    days_since_last_whatever = float(cw_timestamp - lw_timestamp) / (60 * 60 * 24)
    if days_since_last_whatever > CONFIG.CONFIG_DAYS_SINCE_LAST_WORKOUT_CAP:
        days_since_last_whatever = CONFIG.CONFIG_DAYS_SINCE_LAST_WORKOUT_CAP

    return days_since_last_whatever


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
        velocities_m_per_s_arr,
        set_number_of_the_day

):
    packaged_workout = {}

    # use a hot vector for which exercise they r doing
    for iii in range(len(global_exercise_vocabulary)):
        packaged_workout["category_exercise_name_" + str(iii)] = 0
    ex_name = exercise_name

    if ex_name is not -1:
        en = global_exercise_vocabulary.index(ex_name)
        packaged_workout["category_exercise_name_" + str(en)] = 1

    packaged_workout["reps_planned"] = reps_planned

    if reps_completed > CONFIG.CONFIG_MAX_REPS_PER_SET:
        reps_completed = CONFIG.CONFIG_MAX_REPS_PER_SET
    #do not cap min reps_completed bc it would be zero if they fail

    packaged_workout["reps_completed"] = reps_completed
    packaged_workout["weight_lbs"] = weight_lbs

    packaged_workout["postset_heartrate"] = postset_heartrate
    packaged_workout["went_to_failure"] = went_to_failure

    packaged_workout["did_pull_muscle"] = did_pull_muscle

    packaged_workout["used_lifting_gear"] = used_lifting_gear

    packaged_workout["days_since_last_workout"] = unit_days_arr_human[-1]["days_since_last_workout"]

    packaged_workout["day_number"] = len(unit_days_arr_human)

    # pad reps array to a fixed 20 reps
    # that way you can just include a 20 feature array
    # then let the flow go unmessed with

    vmpsa = velocities_m_per_s_arr
    velarr = []

    if reps_completed == 0:
        #so for completly failed reps they have no velocity
        #so they do not go into the set that the stressmodel trains on
        #so the stressmodel doesnt see them
        #need to mod it so that the stress model can be exposed to these
        #ones to see the outcome so if completed reps = 0
        #then for velocity sub in zeroes
        vmpsa = [0.0]



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
            velarr.append(velarr[-1])

    if len(velarr) > CONFIG.CONFIG_MAX_REPS_PER_SET:
        warnings.warn("too many velocities, bad data?")
        velarr = velarr[0:CONFIG.CONFIG_MAX_REPS_PER_SET]

    while len(velarr) < CONFIG.CONFIG_MAX_REPS_PER_SET:
        velarr.append(0)



    for c in range(len(velarr)):
        v = velarr[c]
        if v < 0.0:
            warnings.warn("there exists a negative velocity, bad data?")
            velarr[c]=0.0

    for iiii in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
        packaged_workout["velocities_arr_" + str(iiii)] = velarr[iiii]

    packaged_workout["set_number_of_the_day"] = set_number_of_the_day

    return packaged_workout


# ---------------------------------------------------------->



def get_norm_values():
    #cache the values dont keep reloading
    if get_norm_values.GLOBAL_NORM_VALS is None:
        get_norm_values.GLOBAL_NORM_VALS = pickle.load(open(CONFIG.CONFIG_NORMALIZE_VALS_PATH, "rb"))
    return get_norm_values.GLOBAL_NORM_VALS
get_norm_values.GLOBAL_NORM_VALS = None

def write_norm_values():

    stress_picklefilenames = get_stress_unit_names()
    body_picklefilenames = get_body_unit_names()

    # stress model data and body data are the same
    # so it doesn't matter which one you derive the norm values from

    a_stress_unpickled_package = pickle.load(open(CONFIG.CONFIG_NN_STRESS_MODEL_PICKLES_PATH + stress_picklefilenames[0], "rb"))

    dayseriesxmin = [9999.0] * len((a_stress_unpickled_package["dayseriesx"][0]).keys())
    dayseriesxmax = [-9999.0] * len((a_stress_unpickled_package["dayseriesx"][0]).keys())

    userxmin = [9999.0] * len((a_stress_unpickled_package["userx"]).keys())
    userxmax = [-9999.0] * len((a_stress_unpickled_package["userx"]).keys())

    workoutxseriesmin = [9999.0] * len((a_stress_unpickled_package["workoutxseries"][0]).keys())
    workoutxseriesmax = [-9999.0] * len((a_stress_unpickled_package["workoutxseries"][0]).keys())

    workoutymin = [9999.0] * len((a_stress_unpickled_package["workouty"]).keys())
    workoutymax = [-9999.0] * len((a_stress_unpickled_package["workouty"]).keys())

    a_body_unpickled_package = pickle.load(
        open(CONFIG.CONFIG_NN_BODY_MODEL_PICKLES_PATH + body_picklefilenames[0], "rb"))

    dayymin = [9999.0] * len((a_body_unpickled_package["dayy"]).keys())
    dayymax = [-9999.0] * len((a_body_unpickled_package["dayy"]).keys())

    for picklefilename in stress_picklefilenames:

        unpickled_package = pickle.load(open(CONFIG.CONFIG_NN_STRESS_MODEL_PICKLES_PATH + picklefilename, "rb"))

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

    for picklefilename in body_picklefilenames:
        unpickled_package = pickle.load(open(CONFIG.CONFIG_NN_BODY_MODEL_PICKLES_PATH + picklefilename, "rb"))
        unpickleddayy = unpickled_package["dayy"]
        unpickleddayykeys = sorted(list((unpickleddayy.keys())))
        for i in range(len(unpickleddayykeys)):
            akey = unpickleddayykeys[i]
            aval = float(unpickleddayy[akey])
            if aval < dayymin[i]:
                dayymin[i] = aval
            if aval > dayymax[i]:
                dayymax[i] = aval


    normVals = {}
    normVals["dayseriesxmin"] = dayseriesxmin
    normVals["daysseriesxmax"] = dayseriesxmax
    normVals["userxmin"] = userxmin
    normVals["userxmax"] = userxmax
    normVals["workoutxseriesmin"] = workoutxseriesmin
    normVals["workoutxseriesmax"] = workoutxseriesmax
    normVals["workoutymin"] = workoutymin
    normVals["workoutymax"] = workoutymax
    normVals["dayymin"] = dayymin
    normVals["dayymax"] = dayymax

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
    dayymin = norm_vals["dayymin"]
    dayymax = norm_vals["dayymax"]

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


    if "workouty" in packaged_unit.keys():
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

    if "dayy" in packaged_unit.keys():
        unpickleddayy = packaged_unit["dayy"]
        n_unpickleddayy = []
        unpickleddayykeys = sorted(list(unpickleddayy.keys()))
        for i in range(len(unpickleddayykeys)):
            akey = unpickleddayykeys[i]
            aval = unpickleddayy[akey]
            if aval > dayymax[i]:
                aval = dayymax[i]
            if aval < dayymin[i]:
                aval = dayymin[i]
            if (dayymax[i]-dayymin[i]) > 0:
                aval = (aval - dayymin[i])/(dayymax[i]-dayymin[i])
            else:
                aval = 0
            n_unpickleddayy.append(aval)
        n_packaged_unit["dayy"] = n_unpickleddayy



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

def make_h_day_with_xh_ym(last_day_xh,day_ym):

    norm_vals = pickle.load(open(CONFIG.CONFIG_NORMALIZE_VALS_PATH, "rb"))

    dayseriesxmin = norm_vals["dayseriesxmin"]
    dayseriesxmax = norm_vals["daysseriesxmax"]
    userxmin = norm_vals["userxmin"]
    userxmax = norm_vals["userxmax"]
    workoutxseriesmin = norm_vals["workoutxseriesmin"]
    workoutxseriesmax = norm_vals["workoutxseriesmax"]
    workoutymin = norm_vals["workoutymin"]
    workoutymax = norm_vals["workoutymax"]
    dayymin = norm_vals["dayymin"]
    dayymax = norm_vals["dayymax"]

    new_h_day = copy.deepcopy(last_day_xh)

    ykeys =["withings_weight_lbs","withings_muscle_mass_percent","withings_body_water_percent","withings_heart_rate_bpm"
            ,"withings_bone_mass_percent","withings_pulse_wave_velocity_m_per_s"]

    ykeys = sorted(list(ykeys))

    for ii in range(len(ykeys)):
        key = ykeys[ii]
        aval = day_ym[ii]
        amin = dayymin[ii]
        amax = dayymax[ii]
        unnormal = (aval * (amax- amin)) + amin
        new_h_day[key] = unnormal

    return new_h_day


def get_stress_unit_names():
    picklefilenames = get_dir_names(CONFIG.CONFIG_NN_STRESS_MODEL_PICKLES_PATH)
    return picklefilenames

def get_body_unit_names():
    picklefilenames = get_dir_names(CONFIG.CONFIG_NN_BODY_MODEL_PICKLES_PATH)
    return picklefilenames

def get_human_unit_for_name(unit_name,path):
    unpickled_package = pickle.load(open(path + unit_name, "rb"))
    return unpickled_package

def get_machine_unit_for_name(unit_name,norm_vals,path):
    human_unit = get_human_unit_for_name(unit_name,path)
    machine_unit = normalize_unit(human_unit, norm_vals)
    return machine_unit

def convert_human_unit_to_machine(h_unit, norm_vals):
    m_unit = normalize_unit(h_unit, norm_vals)
    return m_unit



class Lift_NN():
    def __init__(self, CHOSEN_BATCH_SIZE):

        all_names = get_stress_unit_names()
        norm_vals = get_norm_values()
        loaded_unit = get_machine_unit_for_name(all_names[0], norm_vals, CONFIG.CONFIG_NN_STRESS_MODEL_PICKLES_PATH)

        a_dayseriesx = loaded_unit["dayseriesx"]
        a_userx = loaded_unit["userx"]
        a_workoutxseries = loaded_unit["workoutxseries"]
        a_workouty = loaded_unit["workouty"]

        all_body_names = get_body_unit_names()
        body_loaded_unit = get_machine_unit_for_name(all_body_names[0], norm_vals,
                                                     CONFIG.CONFIG_NN_BODY_MODEL_PICKLES_PATH)
        a_dayy = body_loaded_unit["dayy"]


        with tf.variable_scope('body_model'):
            self.BODY_NUM_Y_OUTPUT = len(a_dayy)

            self.body_day_series_input = tf.placeholder(tf.float32, (None, None, len(a_dayseriesx[0])),
                                                         name="body_day_series_input")
            self.body_workout_series_input = tf.placeholder(tf.float32, (None, None, len(a_workoutxseries[0])),
                                                             name="body_workout_series_input")

            self.body_user_vector_input = tf.placeholder(tf.float32, (None, len(a_userx)),
                                                          name="body_user_vector_input")
            self.body_dayy = tf.placeholder(tf.float32, (None, len(a_dayy)), name="body_dayy")

            with tf.variable_scope('body_workout_series_stageA'):
                body_wo_cellA = tf.contrib.rnn.LSTMCell(100)
                body_wo_rnn_outputsA, body_wo_rnn_stateA = tf.nn.dynamic_rnn(body_wo_cellA,
                                                                               self.body_workout_series_input,
                                                                               dtype=tf.float32)
                body_wo_batchA = tf.layers.batch_normalization(body_wo_rnn_outputsA)

            with tf.variable_scope('body_workout_series_stageB'):
                body_wo_cellB = tf.contrib.rnn.LSTMCell(100)
                body_wo_resB = tf.contrib.rnn.ResidualWrapper(body_wo_cellB)
                body_wo_rnn_outputsB, body_wo_rnn_stateB = tf.nn.dynamic_rnn(body_wo_resB, body_wo_rnn_outputsA,
                                                                               dtype=tf.float32)
                body_wo_batchB = tf.layers.batch_normalization(body_wo_rnn_outputsB)

            with tf.variable_scope('body_workout_series_stageC'):
                body_wo_cellC = tf.contrib.rnn.LSTMCell(100)
                body_wo_resC = tf.contrib.rnn.ResidualWrapper(body_wo_cellC)
                body_wo_rnn_outputsC, world_wo_rnn_stateB = tf.nn.dynamic_rnn(body_wo_resC, body_wo_rnn_outputsB,
                                                                               dtype=tf.float32)
                body_wo_batchC = tf.layers.batch_normalization(body_wo_rnn_outputsC)


            with tf.variable_scope('body_day_series_stageA'):
                body_day_cellA = tf.contrib.rnn.LSTMCell(100)
                body_day_rnn_outputsA, body_day_rnn_stateA = tf.nn.dynamic_rnn(body_day_cellA,
                                                                                 self.body_day_series_input,
                                                                                 dtype=tf.float32)
                body_day_batchA = tf.layers.batch_normalization(body_day_rnn_outputsA)

            with tf.variable_scope('body_day_series_stageB'):
                body_day_cellB = tf.contrib.rnn.LSTMCell(100)
                body_day_resB = tf.contrib.rnn.ResidualWrapper(body_day_cellB)
                body_day_rnn_outputsB, world_day_rnn_stateB = tf.nn.dynamic_rnn(body_day_resB, body_day_rnn_outputsA,
                                                                                 dtype=tf.float32)
                body_day_batchB = tf.layers.batch_normalization(body_day_rnn_outputsB)

            with tf.variable_scope('body_day_series_stageC'):
                body_day_cellC = tf.contrib.rnn.LSTMCell(100)
                body_day_resC = tf.contrib.rnn.ResidualWrapper(body_day_cellC)
                body_day_rnn_outputsC, body_day_rnn_stateC = tf.nn.dynamic_rnn(body_day_resC, body_day_rnn_outputsB,
                                                                                 dtype=tf.float32)
                body_day_batchC = tf.layers.batch_normalization(body_day_rnn_outputsC)


            body_lastA = body_wo_rnn_outputsC[:, -1:]  # get last lstm output
            body_lastAA = body_day_rnn_outputsC[:, -1:]  # get last lstm output

            # world_lastA = world_wo_batchB[:, -1:]  # get last lstm output
            # world_lastAA = world_day_batchB[:, -1:]  # get last lstm output


            # world_lastA = world_wo_rnn_outputsA[:, -1:]  # get last lstm output
            # world_lastAA = world_day_rnn_outputsAA[:, -1:]  # get last lstm output

            self.body_lastA = body_lastA
            self.body_lastAA = body_lastAA

            # takes those two 250 and concats them to a 500
            self.body_combined = tf.concat([body_lastA, body_lastAA], 2)
            self.body_b4shape = tf.shape(self.body_combined)

            # so at setup time you need to know the shape
            # otherwise it is none
            # and the dense layer cannot be setup with a none dimension
            self.body_combined_shaped = tf.reshape(self.body_combined, (CHOSEN_BATCH_SIZE, 100 + 100))
            self.body_afshape = tf.shape(self.body_combined_shaped)
            # tf.set_shape()

            self.body_combined2 = tf.concat([self.body_combined_shaped, self.body_user_vector_input], 1)
            body_dd = tf.layers.dense(self.body_combined2, self.BODY_NUM_Y_OUTPUT)

            self.body_y = body_dd

            self.body_e = tf.losses.mean_squared_error(self.body_dayy, self.body_y)
            self.body_operation = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.body_e)





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


def build_batch_from_names(batch_unit_names, batch_size,path, for_human=None):
    assert for_human is not None, "forgot to specify for_human flag build_batch_from_names"

    day_series_batch = []
    user_x_batch = []
    workout_series_batch = []
    y_batch = []

    norm_vals = get_norm_values()
    i = 0
    while i < batch_size:
        i = i + 1
        a_name = batch_unit_names.pop(0)

        loaded_unit = None
        if for_human:
            loaded_unit = get_human_unit_for_name(a_name, path)
        else:
            loaded_unit = get_machine_unit_for_name(a_name, norm_vals, path)

        a_day_series = loaded_unit["dayseriesx"]
        a_user_x = loaded_unit["userx"]
        a_workout_series = loaded_unit["workoutxseries"]
        a_y = None

        if "workouty" in loaded_unit.keys():
            a_y = loaded_unit["workouty"]
        if "dayy" in loaded_unit.keys():
            a_y = loaded_unit["dayy"]

        day_series_batch.append(a_day_series)
        user_x_batch.append(a_user_x)
        workout_series_batch.append(a_workout_series)
        y_batch.append(a_y)

    day_series_batch = np.array(day_series_batch)
    user_x_batch = np.array(user_x_batch)
    workout_series_batch = np.array(workout_series_batch)
    y_batch = np.array(y_batch)

    # print workout_y_batch.shape
    # print workout_series_batch.shape
    # print user_x_batch.shape
    # print day_series_batch.shape

    return y_batch, workout_series_batch, user_x_batch, day_series_batch


def generate_training_data():
    make_raw_units()
    write_norm_values()


def train_body_model():

    all_names = get_body_unit_names()

    alw = Lift_NN(CONFIG.CONFIG_BATCH_SIZE)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)

    latest_checkpoint = tf.train.latest_checkpoint(CONFIG.CONFIG_SAVE_MODEL_LOCATION)
    if latest_checkpoint is not None:
        alw.asaver.restore(sess, latest_checkpoint)

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

            dayy_batch, wo_series_batch, user_x_batch, day_series_batch \
                = build_batch_from_names(batch_unit_train_names, CONFIG.CONFIG_BATCH_SIZE,
                                         CONFIG.CONFIG_NN_BODY_MODEL_PICKLES_PATH ,for_human=False)

            # print "before"
            # print wo_y_batch.shape
            # print wo_series_batch.shape
            # print user_x_batch.shape
            # print day_series_batch.shape

            ABC = None

            train_results = sess.run([
                alw.body_day_series_input,
                alw.body_workout_series_input,
                alw.body_y,
                alw.body_operation,
                alw.body_e,
                alw.body_dayy,
                alw.body_combined,
                alw.body_combined_shaped,
                alw.body_b4shape,
                alw.body_lastA,
                alw.body_lastAA

                # alw.combined2,
                # alw.workout_y,
                # alw.afshape,
                # alw.user_vector_input
            ],

            feed_dict={
                alw.body_day_series_input: day_series_batch,
                alw.body_workout_series_input: wo_series_batch,
                alw.body_user_vector_input: user_x_batch,
                alw.body_dayy: dayy_batch
            })
            abc = None
            train_error += float(train_results[4])
            print "train_extern: " + str(train_error / len(train_names))

        while len(valid_names_copy) > CONFIG.CONFIG_BATCH_SIZE:

            batch_unit_valid_names = valid_names_copy[:CONFIG.CONFIG_BATCH_SIZE]

            for ii in range(CONFIG.CONFIG_BATCH_SIZE):
                valid_names_copy.pop(0)

            day_series_batch = []
            user_x_batch = []
            workout_series_batch = []
            workout_y_batch = []

            dayy_batch, wo_series_batch, user_x_batch, day_series_batch \
                = build_batch_from_names(batch_unit_valid_names, CONFIG.CONFIG_BATCH_SIZE,
                                         CONFIG.CONFIG_NN_BODY_MODEL_PICKLES_PATH, for_human=False)

            # print workout_y_batch.shape
            # print workout_series_batch.shape
            # print user_x_batch.shape
            # print day_series_batch.shape

            ABC = None

            valid_results = sess.run([
                alw.body_day_series_input,
                alw.body_workout_series_input,
                alw.body_y,
                # alw.world_operation,
                alw.body_e,
                alw.body_dayy,
                alw.body_combined,
                alw.body_combined_shaped,
                alw.body_b4shape,
                alw.body_lastA,
                alw.body_lastAA

                # alw.combined2,
                # alw.workout_y,
                # alw.afshape,
                # alw.user_vector_input
            ],

                feed_dict={
                    alw.body_day_series_input: day_series_batch,
                    alw.body_workout_series_input: wo_series_batch,
                    alw.body_user_vector_input: user_x_batch,
                    alw.body_dayy: dayy_batch
                })
            abc = None
            valid_error += float(valid_results[3])
            print "valid_extern: " + str(valid_error)

        train_error /= float(len(train_names))
        valid_error /= float(len(valid_names))
        print "train_err: " + str(train_error) + " " + "valid_err: " + str(valid_error) + "best_err: " + str(best_error)

        # have to use this until you have enough samples
        # cuz atm 54 samples is not enough to generalize
        # and you need low low error

        if train_error < best_error:
            best_error = train_error
            print "model saved"
            alw.asaver.save(sess, CONFIG.CONFIG_SAVE_MODEL_LOCATION+CONFIG.CONFIG_MODEL_NAME)

            # if train_error > valid_error:
            #    print "model saved"
            #    alw.asaver.save(sess,CONFIG.CONFIG_SAVE_MODEL_LOCATION)

    sess.close()



def train_stress_adaptation_model():

    all_names = get_stress_unit_names()
    norm_vals = get_norm_values()

    alw = Lift_NN(CONFIG.CONFIG_BATCH_SIZE)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)

    latest_checkpoint = tf.train.latest_checkpoint(CONFIG.CONFIG_SAVE_MODEL_LOCATION)
    if latest_checkpoint is not None:
        alw.asaver.restore(sess, latest_checkpoint)

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
                = build_batch_from_names(batch_unit_train_names, CONFIG.CONFIG_BATCH_SIZE,
                                         CONFIG.CONFIG_NN_STRESS_MODEL_PICKLES_PATH,for_human=False)

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
                = build_batch_from_names(batch_unit_valid_names, CONFIG.CONFIG_BATCH_SIZE,
                                         CONFIG.CONFIG_NN_STRESS_MODEL_PICKLES_PATH,for_human=False)

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
            print "validExtern: " + str(valid_error)

        train_error /= float(len(train_names))
        valid_error /= float(len(valid_names))
        print "train_err: " + str(train_error) + " " + "valid_err: " + str(valid_error) + "best_err: " + str(best_error)

        # have to use this until you have enough samples
        # cuz atm 54 samples is not enough to generalize
        # and you need low low error

        if train_error < best_error:
            best_error = train_error
            print "model saved"
            alw.asaver.save(sess, CONFIG.CONFIG_SAVE_MODEL_LOCATION+CONFIG.CONFIG_MODEL_NAME)

            # if train_error > valid_error:
            #    print "model saved"
            #    alw.asaver.save(sess,CONFIG.CONFIG_SAVE_MODEL_LOCATION)

    sess.close()


def train_rl_agent():
    all_names = get_stress_unit_names()
    norm_vals = get_norm_values()

    RL_BATCH_SIZE = 1
    alw = Lift_NN(RL_BATCH_SIZE)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)

    latest_checkpoint = tf.train.latest_checkpoint(CONFIG.CONFIG_SAVE_MODEL_LOCATION)

    if latest_checkpoint is None:
        assert("you need to train the stress model first right now it doesnt exist")

    if latest_checkpoint is not None:
        alw.asaver.restore(sess, latest_checkpoint)


    gradBuffer = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rl_agent'))

    # starting_point_name = []
    # starting_point_name.append(all_names[5])

    most_reward_save_check = -9999

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

            state = {}

            EPISODE_LENGTH = 40

            actions_episode_log_human = []
            reward_log_human = []

            reward_episode = []
            action_index_episode = []
            value_episode = []
            dayseriesx_episode = []
            userx_episode = []
            workoutxseries_episode = []

            # then run the whole set for x epochs (add loop)
            # run an episode with each sample(add looop)

            a_episode_length = EPISODE_LENGTH

            value_episode,reward_episode,action_index_episode,dayseriesx_episode,userx_episode,workoutxseries_episode,\
            actions_episode_log_human,reward_log_human = walk_episode_with_sample(a_sample_name,a_episode_length,sess,norm_vals,alw,state,value_episode,reward_episode,
                                     action_index_episode,dayseriesx_episode,userx_episode,workoutxseries_episode,
                                     actions_episode_log_human,reward_log_human)


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

            #diagnostic
            #for entry in actions_episode_log_human:
            #   print entry
            #print "EPISODE OVER"
            #for entry in reward_log_human:
            #    print entry
            ABC = None


        results2 = sess.run([alw.copy_new_to_old], feed_dict=feed_dict)

        feed_dict = dict(zip(alw.gradient_holders, gradBuffer))
        results1 = sess.run([alw.update_batch], feed_dict=feed_dict)


        rps = np.mean(reward_per_sample)
        reward_per_epoch.append(rps)
        print str(aepoch) + " " + str(rps) + " " + str(np.mean(reward_per_epoch))

        if rps > most_reward_save_check:
            print "saved model"
            most_reward_save_check = rps
            alw.asaver.save(sess, CONFIG.CONFIG_SAVE_MODEL_LOCATION + CONFIG.CONFIG_MODEL_NAME)


    print reward_per_epoch


def walk_episode_with_sample(a_sample_name,
                             a_episode_length,
                             sess,
                             norm_vals,
                            alw,
                            state,
                            value_episode,
                            reward_episode,
                            action_index_episode,
                            dayseriesx_episode,
                            userx_episode,
                             workoutxseries_episode,
                            actions_episode_log_human,
                            reward_log_human):

    a_sample_name_batch = [a_sample_name]

    for i in range(a_episode_length):
        h_unit = {}

        if len(state.keys()) == 0:

            wo_y_batch_h, wo_xseries_batch_h, user_x_batch_h, day_series_batch_h = build_batch_from_names(
                a_sample_name_batch, 1,CONFIG.CONFIG_NN_STRESS_MODEL_PICKLES_PATH, for_human=True)

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

            state["exercises_left"] = copy.deepcopy(CHOOSABLE_EXERCISES)
            state["current_exercise"] = state["exercises_left"][0]
            state["current_weight"] = CONFIG.MINIMUM_WEIGHT
            state["current_reps"] = 6
            state["set_number_of_the_day"] = 0

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

                ex_index = global_exercise_vocabulary.index(exercise_name)
                ex_key = "category_exercise_name_" + str(ex_index)

                last_seen_index = None

                for ik in range(len(h_unit["workoutxseries"])):
                    a_workout_step = h_unit["workoutxseries"][ik]
                    if a_workout_step[ex_key] == 1:
                        last_seen_index = ik

                state["lastrewarddetectedindexes"][exercise_name] = last_seen_index


        else:
            h_unit["dayseriesx"] = state["dayseriesx"]
            h_unit["userx"] = state["userx"]
            h_unit["workoutxseries"] = state["workoutxseries"]
            h_unit["workouty"] = {}
            h_unit["lastrewarddetectedindexes"] = state["lastrewarddetectedindexes"]
            h_unit["exercises_left"] = state["exercises_left"]
            h_unit["current_exercise"] = state["current_exercise"]
            h_unit["current_weight"] = state["current_weight"]
            h_unit["current_reps"] = state["current_reps"]
            h_unit["set_number_of_the_day"] = state["set_number_of_the_day"]


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
            # alw.agent_value1,
            # alw.agent_afshape,
            # alw.agent_combined2,
            alw.agent_user_vector_input,
            # alw.copy_new_to_old,

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


        # picking the random action probability is hard

        # on one hand it encourages exploration
        # on the other hand if you leave it fixed
        # then say you do 10 random actions per 40 length episode
        # you are deviating from the policy by 1/4th
        # which makes the credit assignment problem of
        # determining what action you took that caused the change
        # that much more difficult

        # if you leave it too small then it takes forever

        # so it seems best if you use a decaying random probability
        # or just set it low and train forever


        percent_done = 0.99  # float(aepoch)/float(NUM_EPOCHS)
        #percent_done = 0.10
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
        action = human_readable_action

        if i==0:
            #make RL start with a clean slate, aka on a new day
            action = LEAVE_GYM


        # now pass the chosen action + state to the env

        state, reward, actions_episode_log_human, end_episode, reward_log_human = agent_world_take_step(state, action,
                                                                                                        alw, sess,
                                                                                                        actions_episode_log_human,
                                                                                                        reward_log_human)

        if end_episode:
            break

        value_episode.append(agent_value)
        reward_episode.append(reward)
        action_index_episode.append(action_index)
        dayseriesx_episode.append(m_unit["dayseriesx"][:])
        userx_episode.append(m_unit["userx"][:])
        workoutxseries_episode.append(m_unit["workoutxseries"][:])

    return value_episode,\
           reward_episode,\
           action_index_episode,\
           dayseriesx_episode,\
           userx_episode,workoutxseries_episode, actions_episode_log_human,reward_log_human





def body_model_predict_new_day(a_day_series_h,a_user_x_h,a_workout_series_h,ai_graph,sess):

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
        alw.body_day_series_input,
        alw.body_workout_series_input,
        alw.body_y
    ],

        feed_dict={
            alw.body_day_series_input: day_series_batch,
            alw.body_workout_series_input: workout_series_batch,
            alw.body_user_vector_input: user_x_batch
        })

    m_predicted_day_vals = results_of_action[2][0]
    last_day = a_day_series_h[-1]

    new_h_day = make_h_day_with_xh_ym(last_day,m_predicted_day_vals)

    return new_h_day




def agent_world_add_day(a_day_series_h,a_user_x_h,a_workout_series_h,ai_graph,sess,actions_episode_log_human,state_h,state):
    new_day = body_model_predict_new_day(a_day_series_h, a_user_x_h, a_workout_series_h, ai_graph, sess)

    new_day["day_number"] = new_day["day_number"] + 1

    # look to see how many days in a row we have skipped
    actions_episode_log_human_copy = copy.deepcopy(actions_episode_log_human)
    days_skipped = 0
    did_find_day_skipped = True
    while did_find_day_skipped and len(actions_episode_log_human_copy) > 0:
        entry = actions_episode_log_human_copy[-1]
        check_string = "env " + LEAVE_GYM
        if check_string in entry:
            days_skipped = days_skipped + 1
            actions_episode_log_human_copy.pop()
        else:
            did_find_day_skipped = False

    new_day["days_since_last_workout"] = days_skipped

    # can just take the last day and add it again

    a_day_series_h = np.append(a_day_series_h, copy.deepcopy(new_day))
    a_day_series_h = np.delete(a_day_series_h, 0)
    # ^^remove first item to keep the array size the same

    state_h['dayseriesx'] = a_day_series_h
    state_h['userx'] = state['userx']
    state_h['workoutxseries'] = state['workoutxseries']

    # need this cuz we want the rl to learn across workouts
    # and doing next exercise depletes this so you cannot
    # do LEAVEGYM and keep going unless you replenish
    state_h["exercises_left"] = copy.deepcopy(CHOOSABLE_EXERCISES)

    state_h["current_exercise"] = state_h["exercises_left"][0]
    state_h["current_weight"] = CONFIG.MINIMUM_WEIGHT
    state_h["current_reps"] = CONFIG.STARTING_REPS
    state_h["set_number_of_the_day"] = 0

    actions_episode_log_human.append("env " + LEAVE_GYM)

    return a_day_series_h,a_user_x_h,a_workout_series_h,ai_graph,sess,actions_episode_log_human,\
           state_h,state

    #print "env LEAVEGYM"



def agent_world_take_step(state, action, ai_graph, sess,actions_episode_log_human,reward_log_human):

    # run through the lift world
    # make a new state
    # check for reward

    # need to parse action and insert it into the liftworld input
    # get the output



    a_day_series_h = state['dayseriesx']
    a_user_x_h = state['userx']
    a_workout_series_h = state['workoutxseries']

    # ----------------------------------------------------------

    before_day_change = copy.deepcopy(a_day_series_h[-1])

    # ----------------------------------------------------------

    state_h = state
    reward = 0

    end_episode = False

    if NEXT_EXERCISE in action:
        if len(state_h["exercises_left"]) == 1:
            end_episode = True
        else:
            state_h["exercises_left"].pop(0)
            state_h["current_exercise"] = state_h["exercises_left"][0]
        state_h["current_weight"] = CONFIG.MINIMUM_WEIGHT
        state_h["current_reps"] = state_h["current_reps"]
        actions_episode_log_human.append("env " + NEXT_EXERCISE)

        #sometimes you will see
        #env squat 45.0 6
        #env squat 45.0 6
        #NEXT_EXERCISE
        #env benchpress 55.0 6
        #the reason is after it changed the exercise
        #the next action was to adjust the weight


    if LEAVE_GYM in action:
        a_day_series_h,a_user_x_h,a_workout_series_h,ai_graph,\
        sess,actions_episode_log_human,state_h,state = agent_world_add_day(
            a_day_series_h,a_user_x_h,a_workout_series_h,ai_graph,sess,
                            actions_episode_log_human,state_h,state)




    action_exercise_name_human = state_h["current_exercise"]
    action_planned_weight_human = state_h["current_weight"]
    action_planned_reps_human = state_h["current_reps"]

    #print action_exercise_name_human
    #print action_planned_weight_human
    #print action_planned_reps_human


    if ADJUST_REPS in action:
        rep_adjustment = action.split("=")[1]
        action_planned_reps_human = action_planned_reps_human + float(rep_adjustment)
        if action_planned_reps_human < CONFIG.CONFIG_MIN_REPS_PER_SET:
            #if agent goes out of bounds do nothing to teach it nothing happens
            #when it tries that
            return state_h, reward, actions_episode_log_human, end_episode, reward_log_human
            #action_planned_reps_human = CONFIG.CONFIG_MIN_REPS_PER_SET
        if action_planned_reps_human > CONFIG.CONFIG_MAX_REPS_PER_SET:
            #if agent goes out of bounds do nothing to teach it nothing happens
            #when it tries that
            return state_h, reward, actions_episode_log_human, end_episode, reward_log_human
            #action_planned_reps_human = CONFIG.CONFIG_MAX_REPS_PER_SET

    #print action_planned_reps_human

    if ADJUST_WEIGHT in action:
        weight_adjustment = action.split("=")[1]
        last_weight_lbs = action_planned_weight_human
        action_planned_weight_human = last_weight_lbs + float(weight_adjustment)

    #print action_planned_weight_human


    if LEAVE_GYM not in action and NEXT_EXERCISE not in action:

        # make a workout vector unit from chosen action

        # we will let nn calc rest intervals from
        # times from the start of workout

        exercise_name = action_exercise_name_human
        reps_planned = action_planned_reps_human
        reps_completed = -1

        new_weight_lbs = action_planned_weight_human

        if new_weight_lbs < CONFIG.MINIMUM_WEIGHT:
            #if agent goes out of bounds do nothing to teach it nothing happens
            #when it tries that
            return state_h, reward, actions_episode_log_human, end_episode, reward_log_human
            #new_weight_lbs = CONFIG.MINIMUM_WEIGHT
        if new_weight_lbs > CONFIG.MAXIMUM_WEIGHT:
            #if agent goes out of bounds do nothing to teach it nothing happens
            #when it tries that
            return state_h, reward, actions_episode_log_human, end_episode, reward_log_human
            #new_weight_lbs = CONFIG.MAXIMUM_WEIGHT

        weight_lbs = new_weight_lbs

        #print "env "+exercise_name+" "+str(new_weight_lbs)+" "+str(reps_planned)
        action_log_entry = "env "+exercise_name+" "+str(new_weight_lbs)+" "+str(reps_planned)+" action:"+str(action)
        #print action_log_entry
        actions_episode_log_human.append(action_log_entry)


        postset_heartrate = -1
        went_to_failure = -1
        did_pull_muscle = -1

        used_lifting_gear = -1
        unit_days_arr_human = a_day_series_h
        velocities_m_per_s_arr = -1

        set_number_of_the_day = state_h["set_number_of_the_day"]

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
            velocities_m_per_s_arr,
            set_number_of_the_day
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

        state_h["exercises_left"] = state["exercises_left"]
        state_h["current_exercise"] = state["current_exercise"]
        state_h["current_weight"] = weight_lbs
        state_h["current_reps"] = action_planned_reps_human
        state_h["set_number_of_the_day"] = set_number_of_the_day+1
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
            start_workout_reps_completed = math.floor(start_workout_step["reps_completed"])
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
                start_workout_max_velocity = None
                if int(math.floor(start_workout_reps_completed)) > 0:
                    start_workout_max_velocity = np.amax(start_workout_velocities)
                else:
                    start_workout_max_velocity = 0

                start_workout_force = float(start_workout_max_velocity) * float(start_workout_weight_lbs)

            latest_workout_step = state_h["workoutxseries"][-1]
            latest_workout_reps_completed = math.floor(latest_workout_step["reps_completed"])
            latest_workout_weight_lbs = float(latest_workout_step["weight_lbs"])
            latest_workout_velocities = []
            for iiii in range(int(math.floor(latest_workout_reps_completed))):
                a_velocity = latest_workout_step["velocities_arr_" + str(iiii)]
                latest_workout_velocities.append(a_velocity)

            latest_workout_max_velocity = None
            if int(math.floor(latest_workout_reps_completed)) > 0:
                latest_workout_max_velocity = np.amax(latest_workout_velocities)
            else:
                latest_workout_max_velocity = 0


            latest_workout_force = latest_workout_max_velocity * latest_workout_weight_lbs

            # force here units are in lbs per meters/second oh boy
            # should convert fully to metric but don't really need to
            # bc rl agent doesnt really care about what units
            # reward is in

        reward = 0
        if no_reward_just_set_last_reward_detected_index:
            state_h["lastrewarddetectedindexes"][rl_exercise_chosen_h] = len(state_h["workoutxseries"]) - 1
        else:

            # switching to percentage change for rewards
            # its like normalizing kind results_of_action
            # it makes it easier to make it prioritize diff rewards
            # like gain in muscle mass, loss in body fat, increase in force
            # percentage change kind of normalizes the changes

            # start workout force can be zero when there are zero reps completed
            percent_change = None
            if start_workout_force > 0:
                # looks like you have to have 1% as 1.00 not 0.01
                # i think maybe because discounted returns makes it underflow and disappear otherwise?
                percent_change = (latest_workout_force/start_workout_force)
                percent_change = percent_change - 1.0
                percent_change = percent_change * 100

            else:
                percent_change = 0

            #old way
            #new_reward = latest_workout_force - start_workout_force

            new_reward = percent_change

            if new_reward > 0:
                reward = new_reward
                state_h["lastrewarddetectedindexes"][rl_exercise_chosen_h] = len(state_h["workoutxseries"]) - 1



                start_reward_receipt = str(start_workout_weight_lbs) + " " + str(start_workout_reps_completed) + " " + str(start_workout_max_velocity)
                latest_reward_receipt = str(latest_workout_weight_lbs) + " " + str(latest_workout_reps_completed) + " " + str(latest_workout_max_velocity)
                reward_receipt = str(state_h["current_exercise"]) + " " + str(start_reward_receipt) + " to " + str(latest_reward_receipt)
                reward_receipt = reward_receipt + " " + "(" + str(start_workout_force) + "->" + str(latest_workout_force) + ")"

                reward_log_human.append(reward_receipt)


                # print str(latest_workout_force)+" "+str(start_workout_force)+" "+str(len(state_h["workoutxseries"])-1)+" "+\
                #      str(state_h["lastrewarddetectedindex"])
                # print rl_exercise_chosen_h + " " + str(new_reward)+" "+ str(len(state_h["workoutxseries"])-1) +" : "+str(start_index)
                # print new_reward


        #------------------------------------------------------------------------------------------------------------

        # put in a penalty for each set
        # so the NN encourages the user to leave the gym when
        # it knows doing extra sets will not give a reward
        #
        # no, let it find out that doing less sets gives a better reward
        # through it's interactions with the stress model
        #
        # i dont know if its powerful enough to figure that out tho
        #
        #excessive_sets_penalty = -10
        #reward = reward + excessive_sets_penalty

        #------------------------------------------------------------------------------------------------------------

        # penalize if completed reps is less than planned reps
        # just going to penalize by weight * missed reps bc
        # if you penalize with (the average speed of previous completed reps) * (missed reps)
        # then it gets a free ride when the user fails to complete even 1 rep
        # aka 0 reps completed has no penalty in that case
        # and its cumbersome and questionable to use the avg velocity from a previous
        # set to calc the penalty for this set since there is no velocity
        # when the user gets 0 reps

        #need to figure out an appropriate penalty
        #when doing either of the above
        #it keeps getting negative reward
        #i think the above penalties are too harsh

        #reward is in terms of delta positive force achieved
        #so u need to do the penalty in accordance with that

        # I guess you can let it be a passive penalty
        # bc if they miss all reps then there is no reward

        # I think its ok to just use the missed reps by themselves as
        # a very weak penalty

        #last_workout_step = state_h["workoutxseries"][-1]
        #last_completed_reps = last_workout_step["reps_completed"]
        #last_planned_reps = last_workout_step["reps_planned"]
        #missed_reps_penalty = last_planned_reps - last_completed_reps
        #reward = reward - missed_reps_penalty

        # I thought about this some more
        # it might be ok to just subtract the weight from the reward as a penalty
        # because if all the reps didnt complete maybe it is a 'failed' set

        # after i switched to percentage rewards this wrecks it
        # bc the magnitude is so large compared to percentage
        # lets see if the model can implicitly learn that training to failure is a bad idea?

        # or lets try to convert this penalty to a percentage
        #last_workout_step = state_h["workoutxseries"][-1]
        #last_completed_reps = last_workout_step["reps_completed"]
        #last_planned_reps = last_workout_step["reps_planned"]
        #missed_reps_penalty = float(last_completed_reps)/float(last_planned_reps)
        #missed_reps_penalty = (missed_reps_penalty - 1)*100
        #reward = reward + missed_reps_penalty

        #leave it out for now, try adding it back in later

        #------------------------------------------------------------------------------------------------------------

        # so it gives a probability for pulled muscle
        # so we use that probability to simulate in the environment whether they pull their muscle
        # if pulled muscle do a random percentage with the pulled muscle chance
        # and if it pulls then end the episode


        # this comment may be irrelevant now
        # symptom was a bug I think
        # threshold is it has to be higher than 65%(1std) chance
        # this is bc its really hard for the RL to learn when there is
        # a 10% chance and it triggers
        # it maes it very conservative about picking weights
        # and then it starts picking 45 lbs bc those have the lowest chance

        # but then again using a probability will make it harder for rl agent to learn
        # bc its less deterministic
        # bc you could have the same action sequence twice with different results
        # which will confuse the nn
        # consider doing a fixed percentage, eg 65%?(1STD)


        last_workout_step = state_h["workoutxseries"][-1]
        did_pull_muscle_chance = float(last_workout_step["did_pull_muscle"])

        if did_pull_muscle_chance > 0.95:

            #you can also reduce the episode size as a result of lost days
            #but I think its fine to just let the agent take actions
            #from the new state after x days to recover are added

            if did_pull_muscle_chance > 0.0:
                should_simulate_pulled_muscle = np.random.choice([True,False],p=[did_pull_muscle_chance,1-did_pull_muscle_chance])

                #if should_simulate_pulled_muscle is True:

                days_to_recover_from_pulled_muscle = int(get_num_days_between_pull_recovery())

                for dtr in range(days_to_recover_from_pulled_muscle):
                    a_day_series_h = state_h['dayseriesx']
                    a_user_x = state_h['userx']
                    a_workout_series_h = state_h['workoutxseries']
                    state['workoutxseries'] = a_workout_series_h
                    state['userx'] = a_user_x

                    a_day_series_h, a_user_x_h, a_workout_series_h, ai_graph, \
                    sess, actions_episode_log_human, state_h, state = agent_world_add_day(
                        a_day_series_h, a_user_x_h, a_workout_series_h, ai_graph, sess,
                        actions_episode_log_human, state_h, state)

        #------------------------------------------------------------------------------------------------------------

        #can do the same thing for failure, if fail end the episode
        #no let it figure out failure is bad by itself
        '''
        last_workout_step = state_h["workoutxseries"][-1]
        did_failure_chance = float(last_workout_step["went_to_failure"])
        if did_failure_chance > 0.65:
            #simulate_failure = np.random.choice([True,False],p=[did_failure_chance,1-did_failure_chance])
            #if simulate_failure is True:
            end_episode = True
        '''

        #------------------------------------------------------------------------------------------------------------

        after_day_change = copy.deepcopy(state_h['dayseriesx'][-1])
        if not before_day_change == after_day_change:

            before_bf = before_day_change["withings_body_fat_percent"]
            after_bf = after_day_change["withings_body_fat_percent"]

            before_mm = before_day_change["withings_muscle_mass_percent"]
            after_mm = after_day_change["withings_muscle_mass_percent"]


            bf_percent_change = (after_bf/before_bf)-1.0


            #find out how much it went up
            mm_percent_change = (after_mm/before_mm)-1.0

            body_reward_receipt = "bf:"+str(bf_percent_change)+" mm:"+str(mm_percent_change)
            reward_log_human.append(body_reward_receipt)

            # inverse bf bc we want it to go down
            reward = reward + (bf_percent_change * -1.0) + mm_percent_change



        #----------------------------------------------------------------------------------------------------------

        #reward = reward - 0.001

        ABC = None

    return state_h, reward, actions_episode_log_human,end_episode,reward_log_human



# samples to use
# epochs to use
# receipt

def rl_provide_recommendation_based_on_latest(user_name):

    all_names = get_stress_unit_names()
    norm_vals = get_norm_values()

    user_sample_names = [k for k in all_names if user_name in k]
    all_names = user_sample_names
    all_names = sorted(all_names)

    RL_BATCH_SIZE = 1
    alw = Lift_NN(RL_BATCH_SIZE)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)

    latest_checkpoint = tf.train.latest_checkpoint(CONFIG.CONFIG_SAVE_MODEL_LOCATION)

    if latest_checkpoint is None:
        assert ("you need to train the stress model first right now it doesnt exist")

    if latest_checkpoint is not None:
        alw.asaver.restore(sess, latest_checkpoint)


    a_sample_name = all_names[-1]


    state = {}

    #episode length can be max 300
    #bc after 300 with the 0.99 discounted reward
    #the values pretty much diminish to nothing

    EPISODE_LENGTH = 40

    actions_episode_log_human = []
    reward_log_human = []

    reward_episode = []
    action_index_episode = []
    value_episode = []
    dayseriesx_episode = []
    userx_episode = []
    workoutxseries_episode = []

    # then run the whole set for x epochs (add loop)
    # run an episode with each sample(add looop)

    a_episode_length = EPISODE_LENGTH

    value_episode, reward_episode, action_index_episode, dayseriesx_episode, userx_episode, workoutxseries_episode, \
    actions_episode_log_human, reward_log_human = walk_episode_with_sample(a_sample_name, a_episode_length,
                                                                           sess, norm_vals, alw, state,
                                                                           value_episode, reward_episode,
                                                                           action_index_episode,
                                                                           dayseriesx_episode,
                                                                           userx_episode,
                                                                           workoutxseries_episode,
                                                                           actions_episode_log_human,
                                                                           reward_log_human)

    for action_taken in actions_episode_log_human:
        print action_taken




#generate_training_data()
#train_body_model()
#train_stress_adaptation_model()
#train_rl_agent()
rl_provide_recommendation_based_on_latest("rezahussain")
sys.exit()


GENERATE_DATA_COMMAND = "generate_data"
TRAIN_STRESS_MODEL_COMMAND = "train_stress_model"
TRAIN_RL_AGENT_COMMAND = "train_rl_agent"
RECOMMEND_COMMAND = "recommend"

if len(sys.argv)==1:
    print "Command choices are:"
    print GENERATE_DATA_COMMAND
    print TRAIN_STRESS_MODEL_COMMAND
    print TRAIN_RL_AGENT_COMMAND
    print RECOMMEND_COMMAND + " username_here"
else:
    command = sys.argv[1]

    if command == GENERATE_DATA_COMMAND:
        generate_training_data()
    if command == TRAIN_STRESS_MODEL_COMMAND:
        train_stress_adaptation_model()
    if command == TRAIN_RL_AGENT_COMMAND:
        train_rl_agent()
    if command == RECOMMEND_COMMAND:
        user_name = sys.argv[2]
        rl_provide_recommendation_based_on_latest(user_name)








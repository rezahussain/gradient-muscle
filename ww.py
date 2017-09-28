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


#contact info:
#https://www.facebook.com/reza.hussain.98
#reza@dormantlabs.com


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
    d = open(CONFIG.CONFIG_RAW_JSON_PATH+jsonfilenames[i])
    o = json.load(d)
    jsonobjects.append(o)

#we build one datapoint for each workoutday
#it consists of
#the workout
#then the last workout
#then all of the day vectors on the side in the range
#so we build a range for each datapoint
#based off of the workout lookback


workout_indexes = []

for i in range(len(jsonobjects)):
    jo = jsonobjects[i]
    #print jo['user_vector']
    if len(jo['workout_vector_arr']) > 0:
        workout_indexes.append(i)

abc = None

workout_ranges = []
for i in range(len(workout_indexes)):
    behind = workout_indexes[:i]
    if len(behind) > CONFIG.CONFIG_WORKOUT_LOOKBACK:
        end = workout_indexes[i]
        start = workout_indexes[i-CONFIG.CONFIG_WORKOUT_LOOKBACK]
        workout_ranges.append([start,end])
    #if i-CONFIG_WORKOUT_LOOKBACK >= 0:
    #    workout_ranges.append([i-CONFIG_WORKOUT_LOOKBACK,i])


#---------------------------------------------------------->

#DONE---need to build time vocabulary array
#so from 1-12
#0 to 59
def generate_time_vocabulary():
    times = []
    for i in range(0,24):
        h = i
        suffix = None
        if h<12:
            h = h
            suffix = "am"
        else:
            h = h-12
            suffix = "pm"
        hstr = str(h)
        if len(hstr)<2:
            hstr = "0"+hstr
        if hstr == "00":
            hstr = "12"

        for yy in range(0,60):
            m = yy
            mstr = str(m)
            if len(mstr)<2:
                mstr = "0"+mstr
            times.append(hstr+":"+mstr+suffix)
    return times
time_vocabulary = generate_time_vocabulary()

time_vocabulary.append(-1)
#print time_vocabulary

#---------------------------------------------------------->
# DONE---need to build a vocabulary for the pulled muscle name

pulled_muscle_vocabulary = []
for i in workout_indexes:
    jo = jsonobjects[i]
    workout_vector = jo["workout_vector_arr"]
    for set in workout_vector:
        if set["pulled_muscle_name"] not in pulled_muscle_vocabulary:
            pulled_muscle_vocabulary.append(set["pulled_muscle_name"])



# ---------------------------------------------------------->
#need to build exercise name vocabulary

exercise_vocabulary = []
for i in workout_indexes:
    jo = jsonobjects[i]
    workout_vector = jo["workout_vector_arr"]
    for set in workout_vector:
        if set["exercise_name"] not in exercise_vocabulary:
            exercise_vocabulary.append(set["exercise_name"])

#---------------------------------------------------------->

#need to find max workout_arr length so we can pad
#all of the workouts to this length
#so we can train in batches
#bc batches need the same shapes

max_workout_range_array_len = 0

for r in workout_ranges:
    workout_range_max_num_sets = 0
    for xx in range(r[0],r[1]+1):
        workout_range_max_num_sets += len(jsonobjects[xx]["workout_vector_arr"])
    if workout_range_max_num_sets > max_workout_range_array_len:
        max_workout_range_array_len = workout_range_max_num_sets

Abc = None


#---------------------------------------------------------->

#now we need to find max range day length so we can pad
#all of the day series units to this length
#again so we can train in batches
#bc batches need the same shape

max_day_range_array_len = 0
for r in workout_ranges:
    rdays = r[1]-r[0]
    if rdays > max_day_range_array_len:
        max_day_range_array_len = rdays

#dk if this is a hack
#bc it says 10 but then 11 shows up in the dataset
max_day_range_array_len += 1

Abc = None


#---------------------------------------------------------->

#weight is in lbs
choosable_exercises = ["squat","benchpress","deadlift"]
#now we need to make all of the combos the RLAgent can pick
rl_all_possible_actions = []
for exercise_name in choosable_exercises:
    exercise_index = exercise_vocabulary.index(exercise_name)
    for x in range(1,CONFIG.CONFIG_MAX_REPS_PER_SET+1):
        for y in range(45,CONFIG.CONFIG_MAX_WEIGHT,5):
            rl_all_possible_actions.append("exercise="+exercise_name+":reps="+str(x)+":weight="+str(y))

rl_all_possible_actions.append("exercise=LEAVEGYM:reps=0:weight=0")

ABC = None

#---------------------------------------------------------->


def calc_days_since_last_workout(current_workout_yyyymmdd,last_workout_yyyymmdd):

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



#---------------------------------------------------------->


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



#---------------------------------------------------------->



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

        for xx in range(r[0],r[1]+1):

            debug_name = jsonfilenames[xx]
            # build day array
            # print xx
            packaged_day = {}
            packaged_day["heart_rate_variability_rmssd"] = jsonobjects[xx]["day_vector"]["heart_rate_variability_rmssd"]
            packaged_day["post_day_wearable_calories_burned"] = jsonobjects[xx]["day_vector"]["post_day_wearable_calories_burned"]
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
            packaged_day["withings_pulse_wave_velocity_m_per_s"] = jsonobjects[xx]["day_vector"]["withings_pulse_wave_velocity_m_per_s"]

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
                    result = float(hours) + (float(minutes)/60.0)
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

            #------------------------------------------------------------

            # if you modify the setup above make sure it is also modded
            # in the pad unit_days below
            unit_days.append(packaged_day)


            #need to do it as one big one
            #bc its easier to make sure that you do not include
            #days that are ahead of the current work set
            #but then still include days that are behind or at the current
            #work set


            if len(jsonobjects[xx]["workout_vector_arr"])>0:

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


                    #------------------------------------------------------------------------


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


                    #------------------------------------------------------------------------

                    #so basically for each set in the workout_vector array
                    #we make a train unit with x and y
                    #in this we hollow out the x
                    #then use the unhollowed out x for the y
                    #so the model can learn to predict the values we hollowed out

                    unit_workout_clone = unit_workouts[:]

                    original = unit_workout_clone[-1]
                    copyx = copy.deepcopy(original)
                    #copyx = unit_workout_clone[-1][:]
                    copyy = copy.deepcopy(copyx)


                    # exercise_name categories
                    # reps
                    # weight_lbs
                    copyx["reps_completed"] = -1  # reps_completed

                    copyx["postset_heartrate"] = -1  # postset_heartrate
                    copyx["went_to_failure"] = -1  # went to failure
                    copyx["did_pull_muscle"] = -1

                    #for iii in range(len(pulled_muscle_vocabulary)):
                    #    copyx["category_pulled_muscle_"+str(iii)] = 0

                    # used_lifting_gear
                    # dayssincelastworkout

                    #init the reps speeds to 0
                    for iiii in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
                        copyx["velocities_arr_"+str(iiii)] = 0

                    unit_workout_clone[-1] = copyx



                    #---------------------------------------------------------------------

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

                    #---------------------------------------------------------------------

                    #now pad the day series to the max range
                    #we have seen, so we can train using batches
                    #bc batches need the same size shapes

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


                    ABC=len(unit_days_padded)
                    DEF=None

                    #---------------------------------------------------------------------


                    # only want to train it with at least one of
                    # the measured exertions
                    # heartrate or speed




                    has_heartrate = True
                    if (
                        copyy["postset_heartrate"]==-1

                    ):
                        has_heartrate = False


                    has_velocities = False
                    for iiii in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
                        if copyy["velocities_arr_" + str(iiii)] > 0.0:
                            has_velocities = True

                    has_valid_y = False

                    if has_velocities and has_heartrate:
                        has_valid_y = True

                        #UPDATE: model gets horrible with either
                        #need to use both or one or the other

                        #not sure how I feel about using either
                        #instead of and
                        #but some lifters will not record hr
                        #might have to branch model out
                        #but training it together gives it a better
                        #understanding even with partial data


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

                    #this is what we r gonna package for the NN
                    workoutxseries = unit_workout_clone_padded[:]
                    workouty = copy.deepcopy(copyyy)
                    dayseries = unit_days_padded[:]
                    userx = userx

                    wholeTrainUnit = {}
                    wholeTrainUnit["workoutxseries"] = workoutxseries
                    wholeTrainUnit["workouty"] = workouty
                    wholeTrainUnit["dayseriesx"] = dayseries
                    wholeTrainUnit["userx"] = userx

                    savename = jsonobjects[xx]["day_vector"]["date_yyyymmdd"]+"_"+str(ii)

                    if has_valid_y:
                        pickle.dump(wholeTrainUnit, open(CONFIG.CONFIG_NN_HUMAN_PICKLES_PATH +savename , "wb"))



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

    dayseriesxmin = [9999.0]*len((unpickled_package["dayseriesx"][0]).keys())
    dayseriesxmax = [-9999.0]*len((unpickled_package["dayseriesx"][0]).keys())

    userxmin = [9999.0]*len((unpickled_package["userx"]).keys())
    userxmax = [-9999.0]*len((unpickled_package["userx"]).keys())

    workoutxseriesmin = [9999.0]*len((unpickled_package["workoutxseries"][0]).keys())
    workoutxseriesmax = [-9999.0]*len((unpickled_package["workoutxseries"][0]).keys())

    workoutymin = [9999.0]*len((unpickled_package["workouty"]).keys())
    workoutymax = [-9999.0]*len((unpickled_package["workouty"]).keys())

    for picklefilename in picklefilenames:

        unpickled_package = pickle.load(open(CONFIG.CONFIG_NN_HUMAN_PICKLES_PATH+picklefilename , "rb"))

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



def normalize_unit(packaged_unit,norm_vals):

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


def make_h_workout_with_xh_ym(workoutstep_xh,workoutstep_ym,days_series_arr_h):

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

    '''
    #first make a hollow object
    exercise_name = workoutstep_xh["exercise_name"]
    reps_planned = workoutstep_xh["reps_planned"]
    reps_completed = -1
    weight_lbs = workoutstep_xh["weight_lbs"]
    
    postset_heartrate = -1
    went_to_failure = -1
    did_pull_muscle = -1
    
    used_lifting_gear = workoutstep_xh["used_lifting_gear"]
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
    '''



    h_workout_timestep = workoutstep_hollow_h

    ykeys = ["reps_completed","postset_heartrate","went_to_failure","did_pull_muscle"]
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


def denormalize_workout_series_individual_timestep(n_workout_timestep,days_series_arr_h):

    norm_vals = pickle.load(open(CONFIG.CONFIG_NORMALIZE_VALS_PATH, "rb"))

    dayseriesxmin = norm_vals["dayseriesxmin"]
    dayseriesxmax = norm_vals["daysseriesxmax"]
    userxmin = norm_vals["userxmin"]
    userxmax = norm_vals["userxmax"]
    workoutxseriesmin = norm_vals["workoutxseriesmin"]
    workoutxseriesmax = norm_vals["workoutxseriesmax"]
    workoutymin = norm_vals["workoutymin"]
    workoutymax = norm_vals["workoutymax"]

    #first make a hollow object
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

def get_machine_unit_for_name(unit_name,norm_vals):
    human_unit = get_human_unit_for_name(unit_name)
    machine_unit = normalize_unit(human_unit,norm_vals)
    return machine_unit

def convert_human_unit_to_machine(h_unit,norm_vals):
    m_unit = normalize_unit(h_unit,norm_vals)
    return m_unit



#a_unit = getUnitForName(all_names[0])

class Lift_NN():
    def __init__(self,a_dayseriesx,a_userx,a_workoutxseries,a_workouty,CHOSEN_BATCH_SIZE):


        ##--------------------------------------------------------------------------------------------------------------

        self.WORLD_NUM_Y_OUTPUT = len(a_workouty)

        #self.day_series_input = tf.placeholder(tf.float32 , (None,len(a_dayseriesx),len(a_dayseriesx[0])),name="day_series_input")
        #self.workout_series_input = tf.placeholder(tf.float32, (None, len(a_workoutxseries), len(a_workoutxseries[0])),
        #                                           name="workout_series_input")

        self.world_day_series_input = tf.placeholder(tf.float32, (None, None, len(a_dayseriesx[0])),
                                               name="world_day_series_input")
        self.world_workout_series_input = tf.placeholder(tf.float32, (None, None, len(a_workoutxseries[0])),
        name = "world_workout_series_input")

        self.world_user_vector_input = tf.placeholder(tf.float32 , (None,len(a_userx)),name="world_user_vector_input")
        self.world_workout_y = tf.placeholder(tf.float32, (None, len(a_workouty)), name="world_workouty")



        with tf.variable_scope('world_workout_series_stageA'):
            world_wo_cellA = tf.contrib.rnn.LSTMCell(100)
            world_wo_rnn_outputsA, world_wo_rnn_stateA = tf.nn.dynamic_rnn(world_wo_cellA, self.world_workout_series_input,
                                                                         dtype=tf.float32)
            world_wo_batchA = tf.layers.batch_normalization(world_wo_rnn_outputsA)

        with tf.variable_scope('world_workout_series_stageB'):
            world_wo_cellB = tf.contrib.rnn.LSTMCell(100)
            world_wo_resB = tf.contrib.rnn.ResidualWrapper(world_wo_cellB)
            world_wo_rnn_outputsB, world_wo_rnn_stateB = tf.nn.dynamic_rnn(world_wo_resB, world_wo_rnn_outputsA,
                                                                     dtype=tf.float32)
            world_wo_batchB = tf.layers.batch_normalization(world_wo_rnn_outputsB)


        with tf.variable_scope('world_day_series_stageA'):
            world_day_cellA = tf.contrib.rnn.LSTMCell(100)
            world_day_rnn_outputsA, world_day_rnn_stateA = tf.nn.dynamic_rnn(world_day_cellA, self.world_day_series_input,
                                                                       dtype=tf.float32)
            world_day_batchA = tf.layers.batch_normalization(world_day_rnn_outputsA)

        with tf.variable_scope('world_day_series_stageB'):
            world_day_cellB = tf.contrib.rnn.LSTMCell(100)
            world_day_resB = tf.contrib.rnn.ResidualWrapper(world_day_cellB)
            world_day_rnn_outputsB, world_day_rnn_stateB = tf.nn.dynamic_rnn(world_day_resB,world_day_rnn_outputsA,
                                                                       dtype=tf.float32)
            world_day_batchB = tf.layers.batch_normalization(world_day_rnn_outputsB)

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

        world_lastA = world_wo_rnn_outputsB[:, -1:]  # get last lstm output
        world_lastAA = world_day_rnn_outputsB[:, -1:]  # get last lstm output

        #world_lastA = world_wo_batchB[:, -1:]  # get last lstm output
        #world_lastAA = world_day_batchB[:, -1:]  # get last lstm output


        #world_lastA = world_wo_rnn_outputsA[:, -1:]  # get last lstm output
        #world_lastAA = world_day_rnn_outputsAA[:, -1:]  # get last lstm output

        self.world_lastA = world_lastA
        self.world_lastAA = world_lastAA

        # takes those two 250 and concats them to a 500
        self.world_combined = tf.concat([world_lastA, world_lastAA], 2)
        self.world_b4shape = tf.shape(self.world_combined)

        #so at setup time you need to know the shape
        #otherwise it is none
        #and the dense layer cannot be setup with a none dimension
        self.world_combined_shaped = tf.reshape(self.world_combined,(CHOSEN_BATCH_SIZE,100+100))
        self.world_afshape = tf.shape(self.world_combined_shaped)
        #tf.set_shape()

        self.world_combined2 = tf.concat([self.world_combined_shaped,self.world_user_vector_input],1)
        world_dd = tf.layers.dense(self.world_combined2,self.WORLD_NUM_Y_OUTPUT)

        self.world_y = world_dd

        self.world_e = tf.losses.mean_squared_error(self.world_workout_y, self.world_y)
        self.world_operation = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.world_e)


        ##--------------------------------------------------------------------------------------------------------------

        ##let the agent pick DONE as an exercise
        ##when the env sees that it moves it to the next day
        ##let the agent pick the exercises too
        #just penalize it if it does a circuit
        # can use the same setup as worldNN but diff output

        self.agent_day_series_input = tf.placeholder(tf.float32, (None, None, len(a_dayseriesx[0])),
                                               name="agent_day_series_input")
        self.agent_workout_series_input = tf.placeholder(tf.float32, (None, None, len(a_workoutxseries[0])),
        name = "agent_workout_series_input")
        self.agent_user_vector_input = tf.placeholder(tf.float32 , (None,len(a_userx)),name="agent_user_vector_input")


        ##--------change the below world to agent
        ##--------then setup the outputs
        ##do rl assembling of inputs later when u start coding the rl environment interaction code

        self.AGENT_NUM_Y_OUTPUT = len(rl_all_possible_actions)

        self.agent_day_series_input = tf.placeholder(tf.float32, (None, None, len(a_dayseriesx[0])),
                                               name="agent_day_series_input")


        self.agent_workout_series_input = tf.placeholder(tf.float32, (None, None, len(a_workoutxseries[0])),
        name = "agent_workout_series_input")

        self.agent_user_vector_input = tf.placeholder(tf.float32 , (None,len(a_userx)),name="agent_user_vector_input")

        with tf.variable_scope('agent_workout_series_stage'):
            agent_cellA = tf.contrib.rnn.LSTMCell(250)
            agent_rnn_outputsA, agent_rnn_stateA = tf.nn.dynamic_rnn(agent_cellA, self.agent_workout_series_input, dtype=tf.float32)

        with tf.variable_scope('agent_day_series_stage'):
            agent_cellAA = tf.contrib.rnn.LSTMCell(250)
            agent_rnn_outputsAA, agent_rnn_stateAA = tf.nn.dynamic_rnn(agent_cellAA, self.agent_day_series_input, dtype=tf.float32)

        agent_lastA = agent_rnn_outputsA[:, -1:]  # get last lstm output
        agent_lastAA = agent_rnn_outputsAA[:, -1:]  # get last lstm output

        self.agent_lastA = agent_lastA
        self.agent_lastAA = agent_lastAA

        # takes those two 250 and concats them to a 500
        self.agent_combined = tf.concat([agent_lastA, agent_lastAA], 2)
        self.agent_b4shape = tf.shape(self.agent_combined)

        #so at setup time you need to know the shape
        #otherwise it is none
        #and the dense layer cannot be setup with a none dimension
        self.agent_combined_shaped = tf.reshape(self.agent_combined,(1,500))
        self.agent_afshape = tf.shape(self.agent_combined_shaped)
        #tf.set_shape()

        self.agent_combined2 = tf.concat([self.agent_combined_shaped,self.agent_user_vector_input],1)
        agent_dd = tf.layers.dense(self.agent_combined2,self.AGENT_NUM_Y_OUTPUT)

        dd3 = tf.contrib.layers.softmax(agent_dd)

        self.agent_y = dd3

        ##--------------------------------------------------------------------------------------------------------------

        self.asaver = tf.train.Saver()


def build_batch_from_names(batch_unit_names,batch_size,for_human=None):

    assert for_human is not None,"forgot to specify for_human flag build_batch_from_names"

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
            loaded_unit = get_machine_unit_for_name(a_name,norm_vals)

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

    #print workout_y_batch.shape
    #print workout_series_batch.shape
    #print user_x_batch.shape
    #print day_series_batch.shape

    return workout_y_batch,workout_series_batch,user_x_batch,day_series_batch


def train_stress_adaptation_model():

    make_raw_units()
    write_norm_values()

    all_names = get_unit_names()
    norm_vals = get_norm_values()
    loaded_unit = get_machine_unit_for_name(all_names[0],norm_vals)

    some_day_series = loaded_unit["dayseriesx"]
    some_user_x = loaded_unit["userx"]
    some_workout_series = loaded_unit["workoutxseries"]
    some_workout_y = loaded_unit["workouty"]
    abc = None
    alw = Lift_NN(some_day_series,some_user_x,some_workout_series,some_workout_y,CONFIG.CONFIG_BATCH_SIZE)
    init_op = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)

    split_index = int(math.floor(float(len(all_names))*0.8))

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


        while len(train_names_copy)>CONFIG.CONFIG_BATCH_SIZE:

            batch_unit_train_names = train_names_copy[:CONFIG.CONFIG_BATCH_SIZE]

            for ii in range(CONFIG.CONFIG_BATCH_SIZE):
                train_names_copy.pop(0)

            #day_series_batch = []
            #user_x_batch = []
            #workout_series_batch = []
            #workout_y_batch = []

            wo_y_batch,wo_series_batch,user_x_batch,day_series_batch \
                = build_batch_from_names(batch_unit_train_names,CONFIG.CONFIG_BATCH_SIZE,for_human=False)

            print "before"
            print wo_y_batch.shape
            print wo_series_batch.shape
            print user_x_batch.shape
            print day_series_batch.shape

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

                                #alw.combined2,
                                #alw.workout_y,
                                #alw.afshape,
                                #alw.user_vector_input
                                 ],

                                feed_dict={
                                        alw.world_day_series_input:day_series_batch,
                                        alw.world_workout_series_input:wo_series_batch,
                                        alw.world_user_vector_input:user_x_batch,
                                        alw.world_workout_y:wo_y_batch
                                        })
            abc = None
            train_error += float(train_results[4])
            print "trainExtern: " + str(train_error/len(train_names))


        while len(valid_names_copy)>CONFIG.CONFIG_BATCH_SIZE:

            batch_unit_valid_names = valid_names_copy[:CONFIG.CONFIG_BATCH_SIZE]

            for ii in range(CONFIG.CONFIG_BATCH_SIZE):
                valid_names_copy.pop(0)

            day_series_batch = []
            user_x_batch = []
            workout_series_batch = []
            workout_y_batch = []

            wo_y_batch,wo_series_batch,user_x_batch,day_series_batch \
                = build_batch_from_names(batch_unit_valid_names,CONFIG.CONFIG_BATCH_SIZE,for_human=False)

            #print workout_y_batch.shape
            #print workout_series_batch.shape
            #print user_x_batch.shape
            #print day_series_batch.shape

            ABC = None

            valid_results = sess.run([
                                alw.world_day_series_input,
                                alw.world_workout_series_input,
                                alw.world_y,
                                #alw.world_operation,
                                alw.world_e,
                                alw.world_workout_y,
                                alw.world_combined,
                                alw.world_combined_shaped,
                                alw.world_b4shape,
                                alw.world_lastA,
                                alw.world_lastAA

                                #alw.combined2,
                                #alw.workout_y,
                                #alw.afshape,
                                #alw.user_vector_input
                                 ],

                                feed_dict={
                                        alw.world_day_series_input:day_series_batch,
                                        alw.world_workout_series_input:wo_series_batch,
                                        alw.world_user_vector_input:user_x_batch,
                                        alw.world_workout_y:wo_y_batch
                                        })
            abc = None
            valid_error += float(valid_results[3])
            #print "trainExtern: " + str(train_error)

        train_error /= float(len(train_names))
        valid_error /= float(len(valid_names))
        print "train_err: "+str(train_error)+" "+"valid_err: "+str(valid_error) + "best_err: "+str(best_error)

        #have to use this until you have enough samples
        #cuz atm 54 samples is not enough to generalize
        #and you need low low error

        if train_error<best_error:
            best_error = train_error
            print "model saved"
            alw.asaver.save(sess, CONFIG.CONFIG_SAVE_MODEL_LOCATION)

        #if train_error > valid_error:
        #    print "model saved"
        #    alw.asaver.save(sess,CONFIG.CONFIG_SAVE_MODEL_LOCATION)

    sess.close()



def train_rl_agent():

    all_names = get_unit_names()
    norm_vals = get_norm_values()

    loaded_unit = get_machine_unit_for_name(all_names[0],norm_vals)
    some_day_series = loaded_unit["dayseriesx"]
    some_user_x = loaded_unit["userx"]
    some_workout_series = loaded_unit["workoutxseries"]
    some_workout_y = loaded_unit["workouty"]
    RL_BATCH_SIZE = 1
    alw = Lift_NN(some_day_series,some_user_x,some_workout_series,some_workout_y,RL_BATCH_SIZE)



    init_op = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)

    #saver = tf.train.import_meta_graph(CONFIG.CONFIG_SAVE_MODEL_LOCATION+".meta")
    alw.asaver.restore(sess, tf.train.latest_checkpoint('/Users/admin/Desktop/tmp/'))

    #alw.asaver.restore(sess, CONFIG.CONFIG_SAVE_MODEL_LOCATION)

    #so lets use each of these real datatpoints as a starting point
    #and let the model progress from there for a fixed number of steps?
    shuffle(all_names)

    starting_point_name = []
    starting_point_name.append(all_names[5])

    wo_y_batch_h, wo_xseries_batch_h, user_x_batch_h, day_series_batch_h = build_batch_from_names(starting_point_name,1,for_human=True)
    print wo_y_batch_h.shape
    print wo_xseries_batch_h.shape
    print user_x_batch_h.shape
    print day_series_batch_h.shape


    h_unit = {}
    h_unit["dayseriesx"] = day_series_batch_h[0]
    h_unit["userx"] = user_x_batch_h[0]
    h_unit["workoutxseries"] = wo_xseries_batch_h[0]
    h_unit["workouty"] = wo_y_batch_h[0]

    m_unit = convert_human_unit_to_machine(h_unit,norm_vals)

    day_series_batch_m = [m_unit["dayseriesx"]]
    user_x_batch_m = [m_unit["userx"]]
    wo_xseries_batch_m = [m_unit["workoutxseries"]]
    wo_y_batch_m = [m_unit["workouty"]]

    ABC = None

    results = sess.run([
        alw.agent_day_series_input,
        alw.agent_workout_series_input,
        alw.agent_y
        ],
        feed_dict={
            alw.agent_day_series_input: day_series_batch_m,
            alw.agent_workout_series_input: wo_xseries_batch_m,
            alw.agent_user_vector_input: user_x_batch_m
        })

    agent_softmax_choices = results[2][0]

    oai = np.argmax(agent_softmax_choices)
    abc = None
    print oai


    #now just use the index of the highest softmax value to lookup the action
    #rl_all_possible_actions
    human_readable_action = rl_all_possible_actions[oai]
    print human_readable_action


    #now pass the chosen action + state to the env
    state = {}
    state['dayseriesx'] = day_series_batch_h[0]
    state['userx'] = user_x_batch_h[0]
    state['workoutxseries'] = wo_xseries_batch_h[0]

    action = human_readable_action

    agent_world_take_step(state,action,alw,sess)





#exercise=squat:reps=6:weight=620

def agent_world_take_step(state,action,ai_graph,sess):

    # run through the lift world
    # make a new state
    # check for reward

    # need to parse action and insert it into the liftworld input
    # get the output

    a_day_series_h = state['dayseriesx']
    a_user_x_h = state['userx']
    a_workout_series_h = state['workoutxseries']

    #----------------------------------------------------------

    #make a workout vector unit from chosen action

    action_exercise_name_human = (action.split(":")[0]).split("=")[1]
    action_planned_reps_human = (action.split(":")[1]).split("=")[1]
    action_weight_lbs_human = (action.split(":")[2]).split("=")[1]

    # we will let nn calc rest intervals from
    # times from the start of workout

    exercise_name = action_exercise_name_human
    reps_planned = action_planned_reps_human
    reps_completed = -1
    weight_lbs = action_weight_lbs_human

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

    #a_workout_series.append(workoutstep_for_predict_h)
    #np array so u cant use append like above
    np.append(a_workout_series_h,workoutstep_for_predict_h)

    #----------------------------------------------------------

    #still need to pad and trim to max lengths
    #ez to do, just remove 1 from the back if you are appending 1
    np.delete(a_workout_series_h,0)

    #----------------------------------------------------------
    #convert state to machine format

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

    #---------------------------------------------------------
    #put through graph
    alw = ai_graph

    #print workout_y_batch.shape
    #print workout_series_batch.shape
    #print user_x_batch.shape
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


    #h_workout_step_missing_pieces =
    #def make_h_workout_with_xh_ym(workoutstep_xh, workoutstep_ym, days_series_arr_h):

    h_filled_workout_step = make_h_workout_with_xh_ym(workoutstep_for_predict_h,m_filled_workout_step,a_day_series_h)

    #h_filled_workout_step = denormalize_workout_series_individual_timestep(m_filled_workout_step,a_day_series_h)

    # print m_filled_workout_step
    # print "trainExtern: " + str(train_error)

    ABC = None

    #----------------------------------------------------------


    '''
    #take the value from the last day vector timestep
    #then in environment steps when you advance the day you need to make sure
    #that you make it correctly
    days_since_last_workout_human = None

    if len(a_day_series) > 0:
        days_since_last_workout_machine = a_day_series[-1][23]#its at the dayseries timestep position 23
        #so we denormalize a partial workout timestep
        #so we can get the human value for this normed value
        temp_timestep = denormalize_workout_series_individual_timestep(a_day_series[-1])
        days_since_last_workout_human = temp_timestep[23]
        #denormalize so it matches with everything else in human readable at this stage
    else:
        days_since_last_workout_human = CONFIG.CONFIG_DAYS_SINCE_LAST_WORKOUT_CAP

    new_workout_vector_timestep[10]  #dayssincelastworkout

    # init the reps speeds to 0
    for rs in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
        new_workout_vector_timestep.append(0)


    machine_workout_vector_timestep = normalize_workout_series_individual_timestep(new_workout_vector_timestep)
    #so now normalize the new workout step and add it to the state
    '''



    #----------------------------------------------------------

    '''
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
            alw.world_workout_series_input: workout_series_batch,
            alw.world_user_vector_input: user_x_batch,
            alw.world_workout_y: workout_y_batch
        })
    abc = None
    valid_error = valid_results[3]
    # print "trainExtern: " + str(train_error)
    '''



    #print state,action



train_stress_adaptation_model()
#train_rl_agent()









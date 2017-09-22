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


Abc = None


#---------------------------------------------------------->

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



print workout_ranges

def makeRawPackages():

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
            packaged_day = []
            packaged_day.append(jsonobjects[xx]["day_vector"]["heart_rate_variability_rmssd"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["post_day_wearable_calories_burned"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["post_day_calories_in"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["post_day_protein_g"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["post_day_carbs_g"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["post_day_fat_g"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["withings_weight_lbs"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["withings_body_fat_percent"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["withings_muscle_mass_percent"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["withings_body_water_percent"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["withings_heart_rate_bpm"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["withings_bone_mass_percent"])
            packaged_day.append(jsonobjects[xx]["day_vector"]["withings_pulse_wave_velocity_m_per_s"])

            sbtap = time_vocabulary.index(jsonobjects[xx]["day_vector"]["sleeptime_bed_time_ampm"])
            packaged_day.append(sbtap)

            srtap = time_vocabulary.index(jsonobjects[xx]["day_vector"]["sleeptime_rise_time_ampm"])
            packaged_day.append(srtap)

            packaged_day.append(jsonobjects[xx]["day_vector"]["sleeptime_efficiency_percent"])

            sarap = time_vocabulary.index(jsonobjects[xx]["day_vector"]["sleeptime_alarm_ring_ampm"])
            packaged_day.append(sarap)

            sasap = time_vocabulary.index(jsonobjects[xx]["day_vector"]["sleeptime_alarm_set_ampm"])
            packaged_day.append(sasap)

            packaged_day.append(jsonobjects[xx]["day_vector"]["sleeptime_snoozed"])

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
            packaged_day.append(sahrs)

            slshrs = getNumericalHour(jsonobjects[xx]["day_vector"]["sleeptime_light_sleep_hrs"])
            packaged_day.append(slshrs)

            sdrhrs = getNumericalHour(jsonobjects[xx]["day_vector"]["sleeptime_deep_rem_hrs"])
            packaged_day.append(sdrhrs)


            #days_since_last_workout = None
            if last_day_vector_workout_day_index is not None:
                current_workout_yyyymmdd = jsonobjects[xx]["day_vector"]["date_yyyymmdd"]
                last_workout_yyyymmdd = jsonobjects[last_day_vector_workout_day_index]["day_vector"]["date_yyyymmdd"]
                days_since_last_workout = calc_days_since_last_workout(current_workout_yyyymmdd, last_workout_yyyymmdd)
            else:
                days_since_last_workout = 0

            packaged_day.append(days_since_last_workout)

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

                    packaged_workout = []

                    #aset = jsonobjects[xx]["workout_vector_arr"][ii]


                    ex_name = jsonobjects[xx]["workout_vector_arr"][ii]["exercise_name"]
                    en = exercise_vocabulary.index(ex_name)
                    packaged_workout.append(en)
                    packaged_workout.append(jsonobjects[xx]["workout_vector_arr"][ii]["reps"])
                    packaged_workout.append(jsonobjects[xx]["workout_vector_arr"][ii]["weight_lbs"])

                    # we will let nn calc rest intervals from
                    # times from the start of workout
                    ptap0 = time_vocabulary.index(jsonobjects[xx]["workout_vector_arr"][0]["postset_time_ampm"])
                    ptap1 = time_vocabulary.index(jsonobjects[xx]["workout_vector_arr"][ii]["postset_time_ampm"])
                    packaged_workout.append(ptap1-ptap0)


                    packaged_workout.append(jsonobjects[xx]["workout_vector_arr"][ii]["intraset_heartrate"])
                    packaged_workout.append(jsonobjects[xx]["workout_vector_arr"][ii]["postset_heartrate"])
                    packaged_workout.append(jsonobjects[xx]["workout_vector_arr"][ii]["went_to_failure"])
                    packaged_workout.append(jsonobjects[xx]["workout_vector_arr"][ii]["did_pull_muscle"])
                    pmn = pulled_muscle_vocabulary.index(jsonobjects[xx]["workout_vector_arr"][ii]["pulled_muscle_name"])
                    packaged_workout.append(pmn)
                    packaged_workout.append(jsonobjects[xx]["workout_vector_arr"][ii]["used_lifting_gear"])

                    # add another variable for days inbetween workouts------------------------------------------------
                    # 1 convert mmddyy to timestamp
                    # 2 do subtraction
                    # 3 do division to get number of days
                    # 4 make latest day start at zero then count up backwards
                    # 5 cap days ago to 4 weeks
                    # make latest day start at zero, then count up backwards aka
                    # days ago
                    # cap days ago to 4 weeks
                    # no I think you have to add the variable to the workout arr
                    # the reason is its unclear on what you do for the dayvector
                    # where a day is not a workout day
                    # here we get the days since last workout
                    # the newest starts at 0, then we count up backwards

                    yy = xx
                    last_workout_day_index = None
                    while last_workout_day_index is None:
                        yy = yy - 1
                        if len(jsonobjects[yy]["workout_vector_arr"]) > 0:
                            last_workout_day_index = yy

                    current_workout_yyyymmdd = jsonobjects[xx]["day_vector"]["date_yyyymmdd"]
                    last_workout_yyyymmdd = jsonobjects[last_workout_day_index]["day_vector"]["date_yyyymmdd"]

                    days_since_last_workout = calc_days_since_last_workout(current_workout_yyyymmdd,last_workout_yyyymmdd)

                    packaged_workout.append(days_since_last_workout)

                    #------------------------------------------------------------------------------------------------


                    #pad reps array to a fixed 20 reps
                    #that way you can just include a 20 feature array
                    #then let the flow go unmessed with

                    vmpsa = jsonobjects[xx]["workout_vector_arr"][ii]["velocities_m_per_s_arr"]
                    velarr = []
                    if vmpsa != -1:
                        velarr.extend(vmpsa)
                    while len(velarr) < CONFIG.CONFIG_MAX_REPS_PER_SET:
                        velarr.append(0)

                    if len(velarr) > CONFIG.CONFIG_MAX_REPS_PER_SET:
                        assert "too many velocities, bad data?"

                    packaged_workout.extend(velarr)
                    for c in velarr:
                        if c < 0.0:
                            assert "there exists a negative velocity, bad data?"


                    unit_workouts.append(packaged_workout)


                    #------------------------------------------------------------------------

                    #so basically for each set in the workout_vector array
                    #we make a train unit with x and y
                    #in this we hollow out the x
                    #then use the unhollowed out x for the y
                    #so the model can learn to predict the values we hollowed out

                    unit_workout_clone = unit_workouts[:]

                    copyx = unit_workout_clone[-1][:]
                    copyy = copyx[:]

                    copyx[0] = copyx[0]  # exercise_name
                    copyx[1] = copyx[1]  # reps
                    copyx[2] = copyx[2]  # weight_lbs
                    copyx[3] = copyx[3]  # rest_interval
                    copyx[4] = -1  # intraset_heartrate
                    copyx[5] = -1  # postset_heartrate
                    copyx[6] = -1  # went to failure
                    copyx[7] = -1  # did_pull_muscle
                    copyx[8] = -1 # pulled_muscle_vocab_index
                    copyx[9] = copyx[9] # used_lifting_gear
                    copyx[10] = copyx[10] # dayssincelastworkout

                    #init the reps speeds to 0
                    for rs in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
                        copyx[11+rs] = 0

                    unit_workout_clone[-1] = copyx

                    #---------------------------------------------------------------------

                    # now pad the unit_workout_timeseries to the max range
                    # we have seen, so make it the length of the workout with the max number
                    # of sets that exists in our whole dataset
                    # this makes it so we can train
                    # batches bc for batches the inputs all have to have
                    # the same shape for some reason

                    unit_workout_clone_padded = unit_workout_clone[:]

                    while len(unit_workout_clone_padded) < max_workout_range_array_len:
                        padx = copyx[:]
                        padx[0] = -1  # exercise_name index
                        padx[1] = -1  # reps
                        padx[2] = -1  # weight_lbs
                        padx[3] = -1  # rest_interval
                        padx[4] = -1  # intraset_heartrate
                        padx[5] = -1  # postset_heartrate
                        padx[6] = 0  # went to failure
                        padx[7] = 0  # did_pull_muscle
                        padx[8] = -1  # pulled_muscle_vocab_index
                        padx[9] = 0   # used_lifting_gear
                        padx[10] = 0  # dayssincelastworkout

                        #init the reps speeds to 0
                        for rs in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
                            padx[11+rs] = 0
                        unit_workout_clone_padded.insert(0, padx)


                    #---------------------------------------------------------------------

                    #now pad the day series to the max range
                    #we have seen, so we can train using batches
                    #bc batches need the same size shapes

                    unit_days_padded = unit_days[:]

                    packaged_day_padded = []
                    packaged_day_padded.append(-1)  # heart_rate_variability_rmssd
                    packaged_day_padded.append(-1)  # post_day_wearable_calories_burned
                    packaged_day_padded.append(-1)  # post_day_calories_in
                    packaged_day_padded.append(-1)  # post_day_protein_g
                    packaged_day_padded.append(-1)  # post_day_carbs_g
                    packaged_day_padded.append(-1)  # post_day_fat_g
                    packaged_day_padded.append(-1)  # withings_weight_lbs
                    packaged_day_padded.append(-1)  # withings_body_fat_percent
                    packaged_day_padded.append(-1)  # withings_muscle_mass_percent
                    packaged_day_padded.append(-1)  # withings_body_water_percent
                    packaged_day_padded.append(-1)  # withings_heart_rate_bpm
                    packaged_day_padded.append(-1)  # withings_bone_mass_percent
                    packaged_day_padded.append(-1)  # withings_pulse_wave_velocity_m_per_s
                    packaged_day_padded.append(-1)  # sleeptime_bed_time_ampm
                    packaged_day_padded.append(-1)  # sleeptime_rise_time_ampm
                    packaged_day_padded.append(-1)  # sleeptime_efficiency_percent
                    packaged_day_padded.append(-1)  # sleeptime_alarm_ring_ampm
                    packaged_day_padded.append(-1)  # sleeptime_alarm_set_ampm
                    packaged_day_padded.append(-1)  # sleeptime_snoozed

                    packaged_day_padded.append(-1)  # sleeptime_awake_hrs
                    packaged_day_padded.append(-1)  # sleeptime_light_sleep_hrs
                    packaged_day_padded.append(-1)  # sleeptime_deep_rem_hrs
                    packaged_day_padded.append(-1)  # days_since_last_workout

                    while len(unit_days_padded) < max_day_range_array_len:
                        unit_days_padded.insert(0, packaged_day_padded)


                    #---------------------------------------------------------------------


                    # only want to train it with at least one of
                    # the measured exertions
                    # heartrate or speed

                    has_heartrate = True
                    if (
                        copyy[4]==-1 and
                        copyy[5]==-1
                    ):
                        has_heartrate = False

                    has_speed = False
                    #init the reps speeds to -1
                    for rs in range(CONFIG.CONFIG_MAX_REPS_PER_SET):
                        if copyy[11+rs] > 0:
                            has_speed = True

                    has_valid_y = False

                    if has_speed and has_heartrate:
                        has_valid_y = True

                        #UPDATE: model gets horrible with either
                        #need to use both or one or the other


                        #not sure how I feel about using either
                        #instead of and
                        #but some lifters will not record hr
                        #might have to branch model out
                        #but training it together gives it a better
                        #understanding even with partial data



                    userjson = jsonobjects[xx]["user_vector"]
                    userx = []
                    userx.append(userjson["genetically_gifted"])
                    userx.append(userjson["years_old"])
                    userx.append(userjson["wrist_width_inches"])
                    userx.append(userjson["ankle_width_inches"])
                    userx.append(userjson["sex_is_male"])
                    userx.append(userjson["height_inches"])
                    userx.append(userjson["harris_benedict_bmr"])

                    #this is what we r gonna package for the NN
                    workoutxseries = unit_workout_clone_padded[:]
                    workouty = copyy[:]
                    dayseries = unit_days_padded[:]
                    userx = userx

                    wholeTrainUnit = {}
                    wholeTrainUnit["workoutxseries"] = workoutxseries
                    wholeTrainUnit["workouty"] = workouty
                    wholeTrainUnit["dayseriesx"] = dayseries
                    wholeTrainUnit["userx"] = userx

                    savename = jsonobjects[xx]["day_vector"]["date_yyyymmdd"]+"_"+str(ii)

                    if has_valid_y:
                        pickle.dump(wholeTrainUnit, open(CONFIG.CONFIG_NN_PICKLES_PATH +savename , "wb"))



def getRawPickleFilenames():
    picklefilenames = os.listdir(CONFIG.CONFIG_NN_PICKLES_PATH)
    if ".DS_Store" in picklefilenames:
        picklefilenames.remove(".DS_Store")
    return picklefilenames

def writeNormValues():

    picklefilenames = getRawPickleFilenames()

    #picklefilenames = os.listdir(CONFIG_NN_PICKLES_PATH)
    #if ".DS_Store" in picklefilenames:
    #    picklefilenames.remove(".DS_Store")

    unpickled_package = pickle.load(open(CONFIG.CONFIG_NN_PICKLES_PATH + picklefilenames[0], "rb"))

    dayseriesxmin = [9999.0]*len(unpickled_package["dayseriesx"][0])
    dayseriesxmax = [-9999.0]*len(unpickled_package["dayseriesx"][0])

    userxmin = [9999.0]*len(unpickled_package["userx"])
    userxmax = [-9999.0]*len(unpickled_package["userx"])

    workoutxseriesmin = [9999.0]*len(unpickled_package["workoutxseries"][0])
    workoutxseriesmax = [-9999.0]*len(unpickled_package["workoutxseries"][0])

    workoutymin = [9999.0]*len(unpickled_package["workouty"])
    workoutymax = [-9999.0]*len(unpickled_package["workouty"])

    for picklefilename in picklefilenames:

        unpickled_package = pickle.load(open(CONFIG.CONFIG_NN_PICKLES_PATH+picklefilename , "rb"))

        unpickledayseriesx = unpickled_package["dayseriesx"]
        for i in range(len(unpickledayseriesx)):
            daystep = unpickledayseriesx[i]
            for ii in range(len(daystep)):
                aval = float(daystep[ii])
                if aval < dayseriesxmin[ii]:
                    dayseriesxmin[ii] = aval
                if aval > dayseriesxmax[ii]:
                    dayseriesxmax[ii] = aval

        unpickleduserx = unpickled_package["userx"]
        for i in range(len(unpickleduserx)):
            aval = float(unpickleduserx[i])
            if aval < userxmin[i]:
                userxmin[i] = aval
            if aval > userxmax[i]:
                userxmax[i] = aval

        unpickledworkoutxseries = unpickled_package["workoutxseries"]
        for i in range(len(unpickledworkoutxseries)):
            workoutstep = unpickledworkoutxseries[i]
            for ii in range(len(workoutstep)):
                aval = float(workoutstep[ii])
                if aval < workoutxseriesmin[ii]:
                    workoutxseriesmin[ii] = aval
                if aval > workoutxseriesmax[ii]:
                    workoutxseriesmax[ii] = aval

        unpickledworkouty = unpickled_package["workouty"]
        for i in range(len(unpickledworkouty)):
            aval = float(unpickledworkouty[i])
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



def normalize_unit(norm_vals, packaged_unit):

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
        for ii in range(len(daystep)):
            if daystep[ii] > dayseriesxmax[ii]:
                daystep[ii] = dayseriesxmax[ii]
            if daystep[ii] < dayseriesxmin[ii]:
                daystep[ii] = dayseriesxmin[ii]
            if (dayseriesxmax[ii] - dayseriesxmin[ii]) > 0:
                daystep[ii] = (daystep[ii] - dayseriesxmin[ii]) / (dayseriesxmax[ii] - dayseriesxmin[ii])
            else:
                daystep[ii] = 0
        n_unpickleddayseriesx.append(daystep)
    n_packaged_unit["dayseriesx"] = n_unpickleddayseriesx

    unpickledworkoutxseries = packaged_unit["workoutxseries"]
    n_unpickledworkoutxseries = []
    for i in range(len(unpickledworkoutxseries)):
        workoutstep = unpickledworkoutxseries[i]
        for ii in range(len(workoutstep)):
            if workoutstep[ii] > workoutxseriesmax[ii]:
                workoutstep[ii] = workoutxseriesmax[ii]
            if workoutstep[ii] < workoutxseriesmin[ii]:
                workoutstep[ii] = workoutxseriesmin[ii]
            if (workoutxseriesmax[ii] - workoutxseriesmin[ii]) > 0:
                workoutstep[ii] = (workoutstep[ii] - workoutxseriesmin[ii]) / (
                workoutxseriesmax[ii] - workoutxseriesmin[ii])
            else:
                workoutstep[ii] = 0
        n_unpickledworkoutxseries.append(workoutstep)
    n_packaged_unit["workoutxseries"] = n_unpickledworkoutxseries

    unpickleduserx = packaged_unit["userx"]
    n_unpickleduserx = []
    for i in range(len(unpickleduserx)):
        if unpickleduserx[i] > userxmax[i]:
            unpickleduserx[i] = userxmax[i]
        if unpickleduserx[i] < userxmin[i]:
            unpickleduserx[i] = userxmin[i]
        if (userxmax[i] - userxmin[i]) > 0:
            unpickleduserx[i] = (unpickleduserx[i] - userxmin[i]) / (userxmax[i] - userxmin[i])
        else:
            unpickleduserx[i] = 0
        n_unpickleduserx.append(unpickleduserx[i])
    n_packaged_unit["userx"] = n_unpickleduserx

    unpickledworkouty = packaged_unit["workouty"]
    n_unpickledworkouty = []
    for i in range(len(unpickledworkouty)):
        if unpickledworkouty[i] > workoutymax[i]:
            unpickledworkouty[i] = workoutymax[i]
        if unpickledworkouty[i] < workoutymin[i]:
            unpickledworkouty[i] = workoutymin[i]
        if (workoutymax[i] - workoutymin[i]) > 0:
            unpickledworkouty[i] = (unpickledworkouty[i] - workoutymin[i]) / (workoutymax[i] - workoutymin[i])
        else:
            unpickledworkouty[i] = 0
        n_unpickledworkouty.append(unpickledworkouty[i])
    n_packaged_unit["workouty"] = n_unpickledworkouty

    return n_packaged_unit


#make a full denormalizer later
def denormalize_workout_series_individual_timestep(n_workout_timestep):

    norm_vals = pickle.load(open(CONFIG.CONFIG_NORMALIZE_VALS_PATH, "rb"))

    dayseriesxmin = norm_vals["dayseriesxmin"]
    dayseriesxmax = norm_vals["daysseriesxmax"]
    userxmin = norm_vals["userxmin"]
    userxmax = norm_vals["userxmax"]
    workoutxseriesmin = norm_vals["workoutxseriesmin"]
    workoutxseriesmax = norm_vals["workoutxseriesmax"]
    workoutymin = norm_vals["workoutymin"]
    workoutymax = norm_vals["workoutymax"]

    workout_step = []
    for ii in range(len(n_workout_timestep)):
        unnormal =(n_workout_timestep*(workoutxseriesmax[ii] - workoutxseriesmin[ii])) + workoutxseriesmin[ii]
        workout_step.append(unnormal)

    return workout_step







#now normalize the raw files
def makeNormalizedPickles():
    picklefilenames = getRawPickleFilenames()
    normVals = pickle.load(open(CONFIG.CONFIG_NORMALIZE_VALS_PATH, "rb"))
    for picklefilename in picklefilenames:
        unpickled_package = pickle.load(open(CONFIG.CONFIG_NN_PICKLES_PATH+picklefilename , "rb"))
        n_unpickled_package =  normalize_unit(normVals, unpickled_package)
        pickle.dump(n_unpickled_package, open(CONFIG.CONFIG_NORMALIZED_NN_PICKLES_PATH+picklefilename, "wb"))



def getUnitNames():
    picklefilenames = os.listdir(CONFIG.CONFIG_NORMALIZED_NN_PICKLES_PATH)
    if ".DS_Store" in picklefilenames:
        picklefilenames.remove(".DS_Store")
    return picklefilenames

def getUnitForName(unit_name):
    unpickled_package = pickle.load(open(CONFIG.CONFIG_NORMALIZED_NN_PICKLES_PATH + unit_name, "rb"))
    return unpickled_package




#a_unit = getUnitForName(all_names[0])

class Lift_NN():
    def __init__(self,a_dayseriesx,a_userx,a_workoutxseries,a_workouty):


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

        with tf.variable_scope('world_workout_series_stage'):
            world_cellA = tf.contrib.rnn.LSTMCell(250)
            world_rnn_outputsA, world_rnn_stateA = tf.nn.dynamic_rnn(world_cellA, self.world_workout_series_input, dtype=tf.float32)

        with tf.variable_scope('world_day_series_stage'):
            world_cellAA = tf.contrib.rnn.LSTMCell(250)
            world_rnn_outputsAA, world_rnn_stateAA = tf.nn.dynamic_rnn(world_cellAA, self.world_day_series_input, dtype=tf.float32)

        world_lastA = world_rnn_outputsA[:, -1:]  # get last lstm output
        world_lastAA = world_rnn_outputsAA[:, -1:]  # get last lstm output

        self.world_lastA = world_lastA
        self.world_lastAA = world_lastAA

        # takes those two 250 and concats them to a 500
        self.world_combined = tf.concat([world_lastA, world_lastAA], 2)
        self.world_b4shape = tf.shape(self.world_combined)

        #so at setup time you need to know the shape
        #otherwise it is none
        #and the dense layer cannot be setup with a none dimension
        self.world_combined_shaped = tf.reshape(self.world_combined,(CONFIG.CONFIG_BATCH_SIZE,500))
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


def buildBatchFromNames(batch_unit_names,batch_size):

    day_series_batch = []
    user_x_batch = []
    workout_series_batch = []
    workout_y_batch = []

    i = 0
    while i < batch_size:
        i = i + 1
        a_name = batch_unit_names.pop(0)
        loaded_unit = getUnitForName(a_name)
        a_day_series = loaded_unit["dayseriesx"]
        a_user_x = loaded_unit["userx"]
        a_workout_series = loaded_unit["workoutxseries"]
        a_workout_y = loaded_unit["workouty"]

        day_series_batch.append(a_day_series)
        user_x_batch.append(a_user_x)
        workout_series_batch.append(a_workout_series)
        workout_y_batch.append(a_workout_y)

        # if you are using batches, they have to have the same shape
        # day_series_batch.append(a_day_series)
        # user_x_batch.append(a_user_x)
        # workout_series_batch.append(a_workout_series)
        # workout_y_batch.append(a_workout_y)

    day_series_batch = np.array(day_series_batch)
    user_x_batch = np.array(user_x_batch)
    workout_series_batch = np.array(workout_series_batch)
    workout_y_batch = np.array(workout_y_batch)

    #print workout_y_batch.shape
    #print workout_series_batch.shape
    #print user_x_batch.shape
    #print day_series_batch.shape

    return workout_y_batch,workout_series_batch,user_x_batch,day_series_batch


def trainStressAdaptationModel():

    makeRawPackages()
    writeNormValues()
    makeNormalizedPickles()

    all_names = getUnitNames()
    loaded_unit = getUnitForName(all_names[0])
    some_day_series = loaded_unit["dayseriesx"]
    some_user_x = loaded_unit["userx"]
    some_workout_series = loaded_unit["workoutxseries"]
    some_workout_y = loaded_unit["workouty"]
    abc = None
    alw = Lift_NN(some_day_series,some_user_x,some_workout_series,some_workout_y)
    init_op = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)

    split_index = int(math.floor(float(len(all_names))*0.8))

    shuffle(all_names)
    train_names = all_names[:split_index]
    valid_names = all_names[split_index:]

    for _ in range(0, CONFIG.CONFIG_NUM_EPOCHS):

        train_names_copy = train_names[:]
        shuffle(train_names_copy)

        valid_names_copy = valid_names[:]

        train_error = None
        valid_error = None

        while len(train_names_copy)>CONFIG.CONFIG_BATCH_SIZE:
            batch_unit_train_names = train_names_copy[:CONFIG.CONFIG_BATCH_SIZE]
            train_names_copy.pop(0)

            #day_series_batch = []
            #user_x_batch = []
            #workout_series_batch = []
            #workout_y_batch = []

            wo_y_batch,wo_series_batch,user_x_batch,day_series_batch \
                = buildBatchFromNames(batch_unit_train_names,CONFIG.CONFIG_BATCH_SIZE)

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
            train_error = train_results[4]
            print "trainExtern: " + str(train_error)

        '''
        while len(valid_names_copy)>CONFIG.CONFIG_BATCH_SIZE:
            batch_unit_valid_names = valid_names_copy[:CONFIG.CONFIG_BATCH_SIZE]
            valid_names_copy.pop(0)

            day_series_batch = []
            user_x_batch = []
            workout_series_batch = []
            workout_y_batch = []

            wo_y_batch,wo_series_batch,user_x_batch,day_series_batch = buildBatchFromNames(batch_unit_valid_names)

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
            valid_error = valid_results[3]
            #print "trainExtern: " + str(train_error)
        print "train_err: "+str(train_error)+" "+"valid_err: "+str(valid_error)

        if train_error>valid_error:
            print "model saved"
            alw.asaver.save(sess,CONFIG.CONFIG_SAVE_MODEL_LOCATION)
        '''


    sess.close()


def trainRLAgent():

    all_names = getUnitNames()
    loaded_unit = getUnitForName(all_names[0])
    some_day_series = loaded_unit["dayseriesx"]
    some_user_x = loaded_unit["userx"]
    some_workout_series = loaded_unit["workoutxseries"]
    some_workout_y = loaded_unit["workouty"]
    alw = Lift_NN(some_day_series,some_user_x,some_workout_series,some_workout_y)

    init_op = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)


    #so lets use each of these real datatpoints as a starting point
    #and let the model progress from there for a fixed number of steps?
    shuffle(all_names)

    starting_point_name = []
    starting_point_name.append(all_names[0])

    day_series_batch = []
    user_x_batch = []
    workout_series_batch = []
    workout_y_batch = []

    workout_y_batch, workout_series_batch, user_x_batch, day_series_batch = buildBatchFromNames(starting_point_name,1)

    print workout_y_batch.shape
    print workout_series_batch.shape
    print user_x_batch.shape
    print day_series_batch.shape

    ABC = None

    results = sess.run([
        alw.agent_day_series_input,
        alw.agent_workout_series_input,
        alw.agent_y,

    ],

        feed_dict={
            alw.agent_day_series_input: day_series_batch,
            alw.agent_workout_series_input: workout_series_batch,
            alw.agent_user_vector_input: user_x_batch
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
    state['workout_series'] = workout_series_batch[0]
    state['user'] = user_x_batch[0]
    state['day_series'] = day_series_batch[0]

    action = human_readable_action

    agent_world_take_step(state,action,alw)







def agent_world_take_step(state,action,ai_graph):

    # run through the lift world
    # make a new state
    # check for reward

    # need to parse action and insert it into the liftworld input
    # get the output

    a_workout_series = state['workout_series']
    a_user_x = state['user']
    a_day_series = state['day_series']

    day_series_batch = [a_workout_series]
    user_x_batch = [a_user_x]
    workout_series_batch = [a_workout_series]

    day_series_batch = np.array(day_series_batch)
    user_x_batch = np.array(user_x_batch)
    workout_series_batch = np.array(workout_series_batch)

    #----------------------------------------------------------

    #make a workout vector unit from chosen action

    action_exercise_name_human = (action.split(":")[0]).split("=")[1]
    action_reps_human = (action.split(":")[1]).split("=")[1]
    action_weight_human = (action.split(":")[2]).split("=")[1]

    # we will let nn calc rest intervals from
    # times from the start of workout

    rest_interval_human = None
    if len(a_workout_series)>0:
        ptap0 = time_vocabulary.index(a_workout_series[0]["postset_time_ampm"])
        ptap1 = time_vocabulary.index(a_workout_series[len(a_workout_series)-1]["postset_time_ampm"])
        rest_interval_human = ptap1-ptap0
    else:
        rest_interval_human = 0


    new_workout_vector_timestep = [None*(11)]

    new_workout_vector_timestep[0] = action_exercise_name_human # exercise_name
    new_workout_vector_timestep[1] = action_reps_human          # reps
    new_workout_vector_timestep[2] = action_weight_human        # weight_lbs
    new_workout_vector_timestep[3] = rest_interval_human        # rest_interval

    new_workout_vector_timestep[4] = -1  # intraset_heartrate
    new_workout_vector_timestep[5] = -1  # postset_heartrate
    new_workout_vector_timestep[6] = -1  # went to failure
    new_workout_vector_timestep[7] = -1  # did_pull_muscle
    new_workout_vector_timestep[8] = -1  # pulled_muscle_vocab_index
    new_workout_vector_timestep[9] = -1  # used_lifting_gear

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



    print state,action




#trainRLAgent()
trainStressAdaptationModel()








    #later unpickle and normalize
    #then train





#so now we make a labeled dataset
        #that is for each workout
        #we make an output for each suggested set and
        #what happened after
        #so
        #--current situation
        #--velocities_m_per_s bottomhalf,tophalf
        #--did pull muscle classification
        #--went to failure classification
        #each workout also needs to calc days since workout

        #this time keep them labeled until normalization
        #and combination?

#workoutarray (num_workouts,workout,set_features)
    #workout array has an array of velocities
    #so we are going to condense it down to two features
    #an average of the top half, and an average of the bottom half
    #that way we can pass it in knowing the shape
    #which tensorflow needs
    #want to calculate the max force of both of those too
        #have to convert lb to kg? yes

#dayarray

#user_vector


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import random
import datetime
import numpy as np
import scipy.optimize as opt
import pandas.io.data as web
import matplotlib.pyplot as plt


# LPPL function
def fit_func(t, parameters):
    # Log-Periodic Power Law (LPPL) function (JLS model function)
    # as for a, b, tc, m, c, w and phi, refer to the papers for the detail.
    # start_time is to decide dt, the term window size of the learning data,
    # not used in this function.
    (a, b, tc, m, c, w, phi, start_time) = parameters
    tm = np.power(tc - t, m)
    return np.exp(a + b*tm + c*tm*np.cos(w*np.log(tc-t)-phi))

# Error function for scipy.optimize.fmin_tnc
def error_func(parameters):
    # TIMESERIES and ACTUAL_VALUES are global variables used for the learning process.
    # they will be overwritten when executing stepwize calculations.
    global MAX_ERROR
    global TIMESERIS
    global ACTUAL_VALUES
    # start_time is to decide dt, the term window size of the learning data.
    # start_time can be fixed by limiting the range of it by [0, 0, true].
    (a, b, tc, m, c, w, phi, start_time) = parameters
    if math.isnan(start_time):
      return MAX_ERROR
    timeseries = TIMESERIES[int(start_time):]
    actual_values = ACTUAL_VALUES[int(start_time):]
    # calculate the mean squared errors of the estimated values
    # the error is measured on the actual_values so that can be used
    estimated_values = [fit_func(t, parameters) for t in timeseries]
    diff = np.divide(np.subtract(estimated_values, actual_values), actual_values)
    mse = np.sum(np.power(diff, 2))/(len(timeseries)-1)
    return mse


# class for a set of parameters, representing a gene for genetic alogorithm
class chromosomes:

    # constructor
    def __init__ (self, limits, parameters=[], verbose=True, evaluation=True):
        # initialization
        self.fitness = False
        self.verbose = verbose
        # variable limits is a list of the preset initial range of the parameters
        self.limits = limits
        self.bounds = [(lower, upper) if restriction else (-float('inf'), float('inf'))
                               for (lower, upper, restriction) in limits]
        #self.bounds = None
        # when variable parameters is not given,
        # assign a random float number in the specified range
        if parameters == []:
            self.parameters = [random.uniform(lower, upper)
                               for (lower, upper, restriction) in limits]
        else:
            self.parameters = parameters
        self.number_of_parameters = len(self.parameters)
        # once the parameters given, evaluate it.
        # namely complete all the evaluation process here in the constructor.
        if evaluation:
          self.evaluate()

    # get the critical time stored in the parametes
    def get_critical_time(self):
        return int(self.parameters[2])

    # get the start time stored in the parametes
    def get_start_time(self):
        return int(self.parameters[7])

    # evaluate the fitness of the parameters
    def evaluate(self):
        global MAX_ERROR
        try:
            # evaluate MSE with the parameter set and the LPPL function
            fitness = error_func(self.parameters)
            if self.verbose:
              print "Initial fitness: {:f}".format(fitness)
            # when initial fitness exceeds MAX_ERROR, immediately returns False
            if fitness >= MAX_ERROR:
              return False
            # the parameter set is used as the initial values to minimize the value of error_func
            # note: this idea comes from https://github.com/jd8001/LPPL and thank his contribution 
            (parameters, nfeval, rc) = opt.fmin_tnc(
                  error_func,
                  self.parameters,
                  fprime = None,
                  approx_grad = True,
                  bounds = self.bounds,
                  messages = 0)
            # store the calibration results
            self.parameters = parameters
            self.number_of_functions_evaluated = nfeval
            self.message = opt.tnc.RCSTRINGS[rc]
            self.fitness = error_func(self.parameters)
            if self.verbose:
              print "Reult          : {:f}".format(self.fitness)
        except:
            return False

    # copy
    def copy(self):
        return chromosomes(self.limits, self.parameters, self.verbose, evaluation=False)

    # crossover the parameter set with that of another chromosomes
    def crossover(self, another):
        new_parameters = list(self.parameters) # copy
        # decide a crossover point
        c = random.randint(1, self.number_of_parameters-1)
        for i in xrange(c, self.number_of_parameters):
            new_parameters[i] = another.parameters[i]
        # return a new evaluated chromosomes
        return chromosomes(self.limits, new_parameters, self.verbose)

    # mate the parameter set with that of another chromosomes
    def mate(self, another):
        new_parameters = list(self.parameters) # copy
        for i in xrange(0, self.number_of_parameters):
            if random.randint(0, 1) == 1:
                new_parameters[i] = another.parameters[i]
        # return a new evaluated chromosomes
        return chromosomes(self.limits, new_parameters, self.verbose)

    # mutate the parameter set
    def mutate(self, num_mutation):
        global MUTATION_RANGE
        new_parameters = list(self.parameters) # copy
        for c in xrange(num_mutation):
            # decide a mutation point
            i = random.randint(0, self.number_of_parameters-1)
            (lower, upper, restriction) = self.limits[i]
            p = self.parameters[i]
            # replace the point by a new value
            if restriction:
              if lower < p and p < upper:
                new_parameters[i] = random.triangular(lower, upper, p)
              else:
                new_parameters[i] = random.uniform(lower, upper)
            else:
              new_parameters[i] = p*(1+(random.random()-0.5)*MUTATION_RANGE)
        # return a new evaluated chromosomes.
        return chromosomes(self.limits, new_parameters, self.verbose)

    # estimate the values by LPPL function with the parameter set
    def estimate(self, timeseries):
        return [fit_func(t, self.parameters) for t in timeseries]

    # make string representation
    def __repr__(self):
        (a, b, tc, m, c, w, phi, start_time) = self.parameters
        if self.fitness:
          info = "fitness: {:4.6f}".format(self.fitness)
        else:
          info = "fitness: N.A.       "
        info += " start:{:4d} CT:{:4d} A:{:4.3f} B:{:3.3f} C:{:3.3f} m:{:1.3f} ω:{:1.3f} φ:{:1.3f}".format(int(start_time), int(tc), a, b, c, m, w, phi)
        return info


# population
class population:

    # constructor
    def __init__(self, ga_parameters, verbose=True):
        (limits, pool_max_size, eliminate_ratio, crossover_ratio, mate_ratio, mutate_ratio, num_mutation_points) = ga_parameters
        self.verbose = verbose
        # range limits for chromosomes
        self.limits = limits
        # chromosomes pool
        self.pool = []
        # pool and evolution settings
        self.max_size = pool_max_size
        self.eliminate_ratio = eliminate_ratio
        self.crossover_ratio = crossover_ratio
        self.mate_ratio = mate_ratio
        self.mutate_ratio = mutate_ratio
        self.num_mutation_points = num_mutation_points
        # fitness results
        self.fitness = []
        # breeds chromosomes
        self.breed()

    # breed chromosomes
    def breed(self):
        # breeds until the number of chromosomes reaches the max size
        for i in xrange(len(self.pool), self.max_size):
            if self.verbose:
              print "Breeding chromosomes: "+str(i+1)+" / "+str(self.max_size)
            self.pool.append(chromosomes(self.limits, [], self.verbose))

    # revaluate the fitness of the chromosomes in the pool
    def revaluate(self):
        new_pool = []
        for x in list(self.pool):
          y = chromosomes(self.limits, self.parametes, self.verbose)
          new_pool.append(y)
        self.pool = new_pool

    # eliminate useless chromosomes
    def eliminate(self):
        size = len(self.pool)
        # check if each parameter is within its range
        for x in list(self.pool):
          flag = False
          for i in xrange(len(x.parameters)):
            (lower, upper, restriction) = self.limits[i]
            if restriction:
              if lower > x.parameters[i] or x.parameters[i] > upper:
                flag = True
                break
          # remove it if a parameter is out of range or if fitness is invalid
          if flag or x.fitness == False:
            self.pool.remove(x)
        # sort all chromosomes in the pool
        self.pool.sort(key = lambda x: x.fitness)
        # delete duplicates
        last_fitness = 0
        for x in list(self.pool):
          if x.fitness == last_fitness:
            self.pool.remove(x)
          last_fitness = x.fitness
        # remove the least performers
        size_limit = int(self.max_size * (1.0-self.eliminate_ratio))
        if len(self.pool) > size_limit:
          self.pool = self.pool[:size_limit]
        if self.verbose:
          print "Survived: "+str(size)+" -> "+str(len(self.pool))+" / "+str(size)

    # crossover
    def crossover(self):
        temp_pool = list(self.pool)
        num_crossover = int(len(temp_pool) * self.crossover_ratio)
        for c in xrange(num_crossover):
          x = temp_pool[random.randint(0, len(temp_pool)-1)]
          temp_pool.remove(x)
          y = temp_pool[random.randint(0, len(temp_pool)-1)]
          temp_pool.remove(y)
          self.pool.append(x.crossover(y))

    # mate
    def mate(self):
        temp_pool = list(self.pool)
        num_mate = int(len(temp_pool) * self.mate_ratio)
        for c in xrange(num_mate):
          x = temp_pool[random.randint(0, len(temp_pool)-1)]
          temp_pool.remove(x)
          y = temp_pool[random.randint(0, len(temp_pool)-1)]
          temp_pool.remove(y)
          self.pool.append(x.mate(y))

    # mutate
    def mutate(self):
        temp_pool = list(self.pool)
        num_mutate = int(len(temp_pool) * self.mutate_ratio)
        for c in xrange(num_mutate):
          x = temp_pool[random.randint(0, len(temp_pool)-1)]
          temp_pool.remove(x)
          self.pool.append(x.mutate(self.num_mutation_points))

    # evolve
    def evolve(self, num_generations):
        global NUM_BEST_PERFORMERS
        for i in xrange(num_generations):
          self.breed()
          self.crossover()
          self.mate()
          self.mutate()
          self.eliminate()
          if self.verbose:
            print "--- ["+str(i+1)+"/"+str(num_generations)+"] current best performers ---"
            for j in self.get_top_performers(min(NUM_TOP_PERFORMERS, len(self.pool))):
              print j
          print "\n"
        return self

    # get the summary performance of the chromosomes in the current pool
    def stats(self):
        fitness = [x.fitness for x in self.pool]
        return [np.amax(fitness), np.amin(fitness), np.mean(fitness)]

    # get the top performing chromosomes
    def get_top_performers(self, num):
        result = []
        self.pool.sort(key = lambda x: x.fitness)
        for i in xrange(min(num, len(self.pool))):
            result.append(self.pool[i]) #.copy())
        return result;

    # get string representation of the pool
    def __repr__(self):
        s = ''
        for x in self.pool:
          s += x.str()+"\n"

# get historical stock price data from yahoo finance
def get_historical_data(ticker, start_date, end_date):
    daily_data = web.get_data_yahoo(ticker, start=start_date, end=end_date)
    num_days = len(daily_data)
    timeseries = range(0, num_days)
    values = [daily_data['Adj Close'][i] for i in xrange(num_days)]
    datetimes = map(lambda tm: datetime.datetime(tm.year, tm.month, tm.day), daily_data.index.tolist())
    return [timeseries, values, datetimes]

# pick up the target data from the all historical data series
def get_learning_data(all_data, learning_end_date, max_term):
    (timeseries, actual_values, datetimes) = all_data
    learning_end_pos = 0
    for dt in datetimes:
      if dt >= learning_end_date:
        break
      learning_end_pos += 1
    learning_start_pos = max(0, learning_end_pos - max_term)
    return (timeseries[learning_start_pos:learning_end_pos],
            actual_values[learning_start_pos:learning_end_pos],
            datetimes[learning_start_pos:learning_end_pos])

def generate_ga_parameters(timeseries,
                           days_to_critical_time = 250,
                           min_term = 60,
                           pool_max_size = 250, #100
                           eliminate_ratio = 0.25,
                           crossover_ratio = 0.125,
                           mate_ratio = 0.125,
                           mutate_probability = 0.10,
                           num_mutation = 3):
    init_a = [1.0, 5.0, False]
    init_b = [0.1, 2.0, False]
    init_tc = [timeseries[-1], timeseries[-1]+days_to_critical_time, False]
    init_m = [0.0, 1.0, True]
    init_c = [-1.0, 1.0, False]
    init_w = [0.1, 2.0, False]
    init_phi = [0.0, np.pi, False]
    init_start = [0, timeseries[-1]-min_term, True]
    limits = (init_a, init_b, init_tc, init_m, init_c, init_w, init_phi, init_start)
    ga_parameters = (limits, pool_max_size, eliminate_ratio, crossover_ratio, mate_ratio, mutate_probability, num_mutation)
    return ga_parameters

# execute the evolution
def execute(generations, ga_parameters, initial_p=False, verbose=True):
    if initial_p:
      # when an existing population is given,
      # the chromosomes in the pool of the population are revaluated
      p = initial_p
      p.revaluate()
    else:
      # otherwise new chromosomes are ganerated and are evaluated.
      p = population(ga_parameters, verbose=verbose)
    p.evolve(generations)
    return p

def single_step(ticker, start_date, end_date, learning_end_date, max_term, min_term, generations, verbose):
    global TIMESERIES
    global ACTUAL_VALUES
    global DATETIMES
    all_data = get_historical_data(ticker, start_date, end_date)
    LEARNING_DATA = get_learning_data(all_data, learning_end_date, max_term)
    (TIMESERIES, ACTUAL_VALUES, DATETIMES) = LEARNING_DATA
    ga_parameters = generate_ga_parameters(TIMESERIES, min_term=min_term)
    p = execute(generations, ga_parameters, verbose)
    draw_single_step(p, all_data, LEARNING_DATA)

def draw_single_step(p, all_data, learning_data):
    global NUM_TOP_PERFORMERS
    (timeseries_all, actual_values_all, datetimes_all) = all_data
    (learning_timeseries, learning_actual_values, learning_datetimes) = learning_data
    learning_end_pos = learning_timeseries[-1]
    # plot the pre-learning and learning part of the actual data
    plt.scatter(timeseries_all[:learning_end_pos],
                actual_values_all[:learning_end_pos], color='black')
    # plot the forecast part
    plt.scatter(timeseries_all[learning_end_pos:],
                actual_values_all[learning_end_pos:], color='blue')
    print "--- RESULT ---"
    c = 0
    for x in p.get_top_performers(NUM_TOP_PERFORMERS):
      print str(x)
      start_time = x.get_start_time()
      pos = learning_timeseries[start_time:][0]
      #ts = learning_timeseries[pos:]
      ts = timeseries_all[pos:]
      # plot the estimations from the start time to the end of learning data
      plt.plot(ts, x.estimate(ts), linewidth=(3 if c==0 else 1))
      c += 1
    plt.show()

def multi_steps(ticker, start_date, end_date, max_term, min_term, prediction_term, generations, verbose):
    global NUM_TOP_PERFORMERS
    global TIMESERIES
    global ACTUAL_VALUES
    global DATETIMES
    # get historical data
    all_data = get_historical_data(ticker, start_date, end_date)
    (timeseries_all, actual_values_all, datetimes_all) = all_data
    # execute multiple steps
    p = False
    results = []
    for learning_end_date in datetimes_all[max_term:]:
      # get learning data for single step execution
      LEARNING_DATA = get_learning_data(all_data, learning_end_date, max_term)
      (TIMESERIES, ACTUAL_VALUES, DATETIMES) = LEARNING_DATA
      ga_parameters = generate_ga_parameters(TIMESERIES, min_term=min_term)
      # execute the ga process
      #p = execute(generations, ga_parameters, initial_p=p, verbose=verbose) #, p)
      p = execute(generations, ga_parameters, verbose=verbose) #, initial_p=False)
      # record results
      results.append((TIMESERIES[-1], p.get_top_performers(NUM_TOP_PERFORMERS)))
      print "--- RESULT ---", DATETIMES[-1]
      for x in p.get_top_performers(NUM_TOP_PERFORMERS):
        print str(x)
    # draw chart
    draw_multi_steps(results, all_data, prediction_term)

def draw_multi_steps(results, all_data, prediction_term):
    (timeseries_all, actual_values_all, datetimes_all) = all_data
    # plot the actual data
    plt.scatter(timeseries_all, actual_values_all, color='black')
    # get band range of the predictions
    best = []
    upper = []
    lower = []
    critical_time = {}
    for (pos, xs) in results:
      r = []
      for x in xs:
        # get and record the prediction
        r.append(x.estimate([pos+prediction_term]))
        # get and record the critical time
        ct = x.get_critical_time()
        if critical_time.has_key(ct):
          critical_time[ct] += 1
        else:
          critical_time[ct] = 1
      if len(xs) > 0:
        # get the best guess
        e = xs[0].estimate([pos+prediction_term])
        best.append((pos, e))
        # get the upper/lower guesses (excluding the extream ones)
        if len(xs) > 3:
          r.sort()
          upper.append((pos, r[len(r)-1-1]))
          lower.append((pos, r[1]))
        else:
          upper.append((pos, e))
          lower.append((pos, e))
    # draw prediction range chart
    c = 0
    #for prediction in (best, upper, lower):
    for prediction in (upper, lower):
      ts = [pos+prediction_term for (pos, e) in prediction]
      pv = [e for (pos, e) in prediction]
      #plt.plot(ts, pv, linewidth=(3 if c==0 else 1))
      plt.plot(ts, pv)
      c += 1
    plt.show()
    # draw critical time distribution
    print critical_time
    keys = critical_time.keys()
    keys.sort()
    vs = []
    for k in keys:
      vs.append(critical_time[k])
    plt.bar(keys, vs)
    plt.show()


MAX_ERROR = 10.0
MUTATION_RANGE = 0.2
NUM_TOP_PERFORMERS = 10

ticker = '^VIX'
start_date = datetime.datetime(2014, 7, 1)
end_date = datetime.datetime(2015, 4, 3)
learning_end_date = datetime.datetime(2014, 4, 3)
max_term = 60
min_term = 20
prediction_term = 2
generations = 10
verbose = False


#single_step(ticker, start_date, end_date, learning_end_date, max_term, min_term, generations, verbose)
multi_steps(ticker, start_date, end_date, max_term, min_term, prediction_term, generations, verbose)

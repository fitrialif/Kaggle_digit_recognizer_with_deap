# import statements needed for the functions below
import random

from deap import base
from deap import creator
from deap import tools
import evaluate # this is done to avoid circular import error


# this function return an odd integer from the range
# range should be even numbers because otherwise distribution of returned number is not even
# it is not really tested what happens if range is negative
def random_odd_int(min_val, max_val):
    integer = random.randint((min_val + 2), (max_val + 1))
    odd_int = int(integer / 2) * 2 - 1
    return odd_int


# this function returns random float rounded to x digits float
def random_rounded_float(min_value, max_value, rounding):
    y = round(random.uniform(min_value, max_value), rounding)
    return y


# this function performs BLX Alpha on a given 2 integer
# as arguments it takes 4 integers () and one float
# it returns a random int from the calculated range
# int1 and int2 are expected to be current values of the gene from parents, int is alpha
# int3 and int4 are maximal ranges of the gene
# this implementation does not behave correctly with negative input
def blx_alpha_int(int1, int2, alpha, int3, int4):
    x = min(int1, int2)
    y = max(int1, int2)
    # range_val = (y - x)
    x_min = int(x - alpha * x)
    if x_min < int3:
        x_min = int3
    y_max = int(y + alpha * y)
    if y_max > int4:
        y_max = int4
    # print(range_val, x, y, x_min, y_max)
    gamma = random.randint(x_min, y_max)
    return gamma


# this function performs BLX Alpha on a given 2 floats
# as arguments it takes 5 floats
# it returns a random float from the calculated range
# float1 and float2 are the values of the gene from the parents 3rd float is alpha
# float3 and float4 are min and max value of the gene
# this implementation does not behave correctly with negative input
def blx_alpha_float(float1, float2, alpha, float3, float4):
    x = min(float1, float2)
    y = max(float1, float2)
    # range_val = (y - x)
    x_min = x - alpha * x
    if x_min < float3:
        x_min = float3
    y_max = y + alpha * y
    if y_max > float4:
        y_max = float4
    # print(range_val, x, y, x_min, y_max)
    gamma = round(random.uniform(x_min, y_max), 3)
    return gamma


# this function performs BLX Alpha on given 2 integers
# as arguments it takes 4 integers () and one float
# it returns a random odd int from the calculated range
# int1 and int2 are expected to be current values of the gene from parents, alpha is float alpha
# int3 and int4 are maximal ranges of the gene
# this implementation does not behave correctly with negative input
def blx_alpha_int_odd(int1, int2, alpha, int3, int4):
    x = min(int1, int2)
    y = max(int1, int2)
    # range_val = (y - x)
    x_min = int(x - alpha * x)
    if x_min < int3:
        x_min = int3
    y_max = int(y + alpha * y)
    if y_max > int4:
        y_max = int4
    # print(range_val, x, y, x_min, y_max)
    # i go easy way just try draw random odd until I succeed
    found = 0
    gamma = 0
    while found == 0:
        gamma = random.randint(x_min, y_max)
        if gamma % 2 != 0:
            found = 1

    return gamma


# this function will return list of attributes needed for the individual
def hyper_parameters(toolbox_):
    hyper_params_ = [toolbox_.attr_batch_size(),
                     toolbox_.attr_learning_rate(),
                     toolbox_.attr_standard_distribution(),
                     toolbox_.attr_receptive_field_conv_1(),
                     toolbox_.attr_filters_conv_1(),
                     toolbox_.attr_receptive_field_conv_2(),
                     toolbox_.attr_filters_conv_2(),
                     toolbox_.attr_activation()]

    return hyper_params_


# GeneRange is a class it contains ranges of all genes as class attributes
class GeneRange(object):
    attr_batch_size_min,             attr_batch_size_max = 10, 10000
    attr_learning_rate_min,          attr_learning_rate_max = 0, 0.4
    attr_standard_distribution_min,  attr_standard_distribution_max = 0, 0.1
    attr_receptive_field_conv_1_min, attr_receptive_field_conv_1_max = 2, 10
    attr_filters_conv_1_min,         attr_filters_conv_1_max = 2, 30
    attr_receptive_field_conv_2_min, attr_receptive_field_conv_2_max = 0, 14
    attr_filters_conv_2_min,         attr_filters_conv_2_max = 1, 30
    attr_activation_min,             attr_activation_max = 0, 7


# register is used to register a function in the toolbox container under an alias
# it may come with some default values for some arguments
# it require toolbox instance and gene_range
def create_toolbox():
    # this internal function is a wrapper for an external one
    # i have to do it because I need function with no arguments/parameters
    def hyper_parameters_():
        x = hyper_parameters(toolbox)
        return x

    # lets create instance of the toolbox
    toolbox = base.Toolbox()

    # lets register attribute generator functions
    toolbox.register("attr_batch_size",             random.randint,
                     GeneRange.attr_batch_size_min,
                     GeneRange.attr_batch_size_max)
    toolbox.register("attr_learning_rate",          random_rounded_float,
                     GeneRange.attr_learning_rate_min,
                     GeneRange.attr_learning_rate_max,
                     3)  # the 3 is rounding
    toolbox.register("attr_standard_distribution",  random_rounded_float,
                     GeneRange.attr_standard_distribution_min,
                     GeneRange.attr_standard_distribution_max,
                     3)  # the 3 is rounding
    toolbox.register("attr_receptive_field_conv_1", random_odd_int,
                     GeneRange.attr_receptive_field_conv_1_min,
                     GeneRange.attr_receptive_field_conv_1_max)
    toolbox.register("attr_filters_conv_1",         random.randint,
                     GeneRange.attr_filters_conv_1_min,
                     GeneRange.attr_filters_conv_1_max)
    toolbox.register("attr_receptive_field_conv_2", random_odd_int,
                     GeneRange.attr_receptive_field_conv_2_min,
                     GeneRange.attr_receptive_field_conv_2_max)  # this range should be even integers
    toolbox.register("attr_filters_conv_2",         random.randint,
                     GeneRange.attr_filters_conv_2_min,
                     GeneRange.attr_filters_conv_2_max)
    toolbox.register("attr_activation",             random.randint,
                     GeneRange.attr_activation_min,
                     GeneRange.attr_activation_max)

    # register additional child classes with use of creator
    # child classes can be reached by calling creator.child class
    # first arg is name of the class, second argument is a parent class,
    # remaining parameters are attributes
    # calling it like this gives pycharm warning that FitnessMax reference cannot be found in the creator.py
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Register functions to create individual and population
    # syntax of this register function/class is:
    # function that instantiate individual class, container and generator
    # where generator is a function that returns list of attributes
    # arguments are just names - they have no brackets ()
    toolbox.register("individual", tools.initIterate, creator.Individual, hyper_parameters_)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register functions to evaluate individual
    from evaluate import evaluate_individual_fast
    from evaluate import evaluate_individual_normal
    from evaluate import evaluate_individual
    toolbox.register("evaluate_fast", evaluate_individual_fast)
    toolbox.register("evaluate_normal", evaluate_individual_normal)
    toolbox.register("evaluate", evaluate_individual)

    # register functions to mate, mutate and select
    toolbox.register("mate", mate_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select_parents", tools.selTournament, tournsize=3)
    toolbox.register("select_elite", tools.selNSGA2)

    return toolbox

# function to mate individuals
# it takes two parents and returns one child
def mate_individuals(toolbox, individual1, individual2):
    # attr_batch_size_min, attr_batch_size_max                         = 10, 10000
    # attr_learning_rate_min, attr_learning_rate_max                   = 0, 1
    # attr_standard_distribution_min, attr_standard_distribution_max   = 0, 1
    # attr_receptive_field_conv_1_min, attr_receptive_field_conv_1_max = 2, 10

    # attr_filters_conv_1_min, attr_filters_conv_1_max                 = 2 ,30
    # attr_receptive_field_conv_2_min, attr_receptive_field_conv_2_max = 0, 14
    # attr_filters_conv_2_min, attr_filters_conv_2_max                 = 1, 30
    # attr_activation_min, attr_activation_max                         = 0, 7

    #     BLXAlphaInt
    #     BLXAlphaIntOdd
    #     BLXAlphaFloat

    child = toolbox.individual()
    alpha = 0.5

    child[0] = blx_alpha_int(individual1[0],     individual2[0], alpha,
                             GeneRange.attr_batch_size_min,
                             GeneRange.attr_batch_size_max)
    child[1] = blx_alpha_float(individual1[1],   individual2[1], alpha,
                               GeneRange.attr_learning_rate_min,
                               GeneRange.attr_learning_rate_max)
    child[2] = blx_alpha_float(individual1[2],   individual2[2], alpha,
                               GeneRange.attr_standard_distribution_min,
                               GeneRange.attr_standard_distribution_max)
    child[3] = blx_alpha_int_odd(individual1[3], individual2[3], alpha,
                                 GeneRange.attr_receptive_field_conv_1_min,
                                 GeneRange.attr_receptive_field_conv_1_max)
    child[4] = blx_alpha_int(individual1[4],     individual2[4], alpha,
                             GeneRange.attr_filters_conv_1_min,
                             GeneRange.attr_filters_conv_1_max)
    child[5] = blx_alpha_int_odd(individual1[5], individual2[5], alpha,
                                 GeneRange.attr_receptive_field_conv_2_min,
                                 GeneRange.attr_receptive_field_conv_2_max)
    child[6] = blx_alpha_int(individual1[6],     individual2[6], alpha,
                             GeneRange.attr_filters_conv_2_min,
                             GeneRange.attr_filters_conv_2_max)
    child[7] = blx_alpha_int(individual1[7],     individual2[7], alpha,
                             GeneRange.attr_activation_min,
                             GeneRange.attr_activation_max)

    return child


#
# TODO: mutate_individual - this function is not yet implemented
def mutate_individual(individual, gene_range):
    # crossover for attributes
    #     batch_size    - average
    #     learning_rate - average
    #     standard_dev  - average
    #     receptive_field_1  - 50% chance to go to the child
    #     filters_1,         - 50% chance to go to the child
    #     receptive_field_2, - 50% chance to go to the child
    #     filters_2,         - 50% chance to go to the child
    #     activation         - 50% chance to go to the child

    # print()

    return individual

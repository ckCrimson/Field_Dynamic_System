import time
from typing import cast

from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField, RealFieldAddition, \
    RealFieldNormTransform, RealFieldValue, RealSingleFieldValue
from one_dim_random_walker.core.states.multi_step_reaching import OneDimensionMultiStepReaching
from one_dim_random_walker.core.states.reachable import OneDimensionReachable
from one_dim_random_walker.core.states.state import IntegerState
from one_dim_random_walker.core.states.state_space import IntegerLine
from one_dim_random_walker.correllation_affected_no_collision.correlation_affected_fds import \
    OneDimRWCorrelatedAffectingFDS
from one_dim_random_walker.correllation_affected_no_collision.correlation_group_evolution import \
    OneDimRWCorrelatedGroupEvolution
from one_dim_random_walker.correllation_affected_no_collision.correlation_grouping import CorrelationAffectingGroups
from one_dim_random_walker.correllation_affected_no_collision.correlation_multi_step import \
    RandomWalkerCorrelatedMultiStepField
from one_dim_random_walker.correllation_affected_no_collision.correlation_reachable import \
    RandomWalkerCorrelatedReachable
from one_dim_random_walker.correllation_affected_no_collision.correlation_single_step import \
    RandomWalkerCorrelatedSingleStep
from one_dim_random_walker.correllation_affected_no_collision.is_correlated import IsRWCorrelated
from one_dim_random_walker.dynamic_systems.dynamic_system import OneDimWalkerFieldDynamicSystem
from one_dim_random_walker.dynamics.multi_step_field import OneDimRandomWalkerMultiStep
from one_dim_random_walker.dynamics.operators import  \
    OneDimRandomWalkerSampleOperator
from one_dim_random_walker.dynamics.single_step import OneDimSingleStep

#------------ Test Parameters ---------------#

number_of_executions = 1
number_of_repeats = 1




#------------- State Test ---------------------#

initial_state = IntegerState(0)

#---------------- State Space Test --------------#

integer_line_1 = IntegerLine(
current_state =  initial_state,
left_limit = -200,
right_limit= 200
)


#-------------Reachable -----------------#

integer_reachable = OneDimensionReachable(6)

#reachable_space = integer_reachable.get_reachable(state=initial_state)

#reachable_space_from_allowed = integer_reachable.get_reachable_from_allowed(state=initial_state,space = integer_line_1)

# ------------Multi Step Reaching ---------------- #

multi_step_reaching = OneDimensionMultiStepReaching(1)
forentier_space = multi_step_reaching.get_multi_step_reaching(initial_state=initial_state,
    reference_space=integer_line_1,
    l=7,  # <-- provide this
    parallel=True  )
#
# for s in forentier_space.get_all_states():
#     print(s)
# 2. Test Code: This is the actual method call you want to time.
# SETUP_CODE = ("from one_dim_random_walker.core.states.multi_step_reaching import OneDimensionMultiStepReaching"
#               "\nfrom one_dim_random_walker.core.states.state import IntegerState"
#               "\nfrom one_dim_random_walker.core.states.state_space import IntegerLine"
#               "\ntest_obj = OneDimensionMultiStepReaching(6)"
#               "\ninitial_state = IntegerState(0)\ninteger_line_1 = IntegerLine(current_state =  initial_state,"
#               "left_limit = -200,"
#               "right_limit= 200)")
# TEST_CODE = "test_obj.get_multi_step_reaching(initial_state, integer_line_1,20)"

# times = timeit.repeat(
#     stmt=TEST_CODE,
#     setup=SETUP_CODE,
#     number=number_of_executions,
#     repeat=number_of_repeats
# )
# best_total_time = min(times)
#
# Calculate the average time per method call
# average_time_per_call = best_total_time / number_of_executions
#
# print(f"Total test time (best of {number_of_repeats} repeats): {best_total_time:.4f} seconds")
# print(f"Average time per call ({number_of_executions} runs): {average_time_per_call * 1000000:.3f} microseconds")

"""

Total test time (best of 5 repeats): 1.7397 seconds
Average time per call (100 runs): 17397.194 microseconds

"""
# -------------  Field  -------------------------  #
field_demo=RealField.from_length(9,initial_state)
#field_demo.plot_field()


#---------------------Field Composition ---------------------#
input_field_1 = RealField.from_length(100,initial_state)
input_field_2 = RealField.from_length(100,initial_state)
addition =   RealFieldAddition()
added_field = addition.apply(input_field_1, input_field_2)

# SETUP_CODE = ("from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField, RealFieldAddition,RealFieldAdditionStateless"
#               "\nfrom one_dim_random_walker.core.states.multi_step_reaching import OneDimensionMultiStepReaching"
#               "\nfrom one_dim_random_walker.core.states.reachable import OneDimensionReachable"
#               "\nfrom one_dim_random_walker.core.states.state import IntegerState"
#               "\nfrom one_dim_random_walker.core.states.state_space import IntegerLine\ninitial_state = IntegerState(0)"
#               "\ninput_field_1 = RealField.from_length(10000,initial_state)\ninput_field_2 = RealField.from_length(20000,initial_state)")
# TEST_CODE = "addition =  RealFieldAddition()\nadded_field = addition.apply(input_field_1, input_field_2)"
# #added_field.plot_field()
# times = timeit.repeat(
#     stmt=TEST_CODE,
#     setup=SETUP_CODE,
#     number=number_of_executions,
#     repeat=number_of_repeats
# )
# best_total_time = min(times)
#
# #Calculate the average time per method call
# average_time_per_call = best_total_time / number_of_executions
# print("Runnig tests over field composition")
# print(f"Total test time (best of {number_of_repeats} repeats): {best_total_time:.4f} seconds")
# print(f"Average time per call ({number_of_executions} runs): {average_time_per_call * 1000000:.3f} microseconds")

#------------------------------------------#

field_norm_transform = RealFieldNormTransform()

current_field=RealField.from_length(100,initial_state)

transformed_field=field_norm_transform.apply(current_field,current_field)
#transformed_field.plot_field()

# SETUP_CODE = ("from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField, RealFieldAddition,RealFieldAdditionStateless,RealFieldNormTransform"
#               "\nfrom one_dim_random_walker.core.states.multi_step_reaching import OneDimensionMultiStepReaching"
#               "\nfrom one_dim_random_walker.core.states.reachable import OneDimensionReachable"
#               "\nfrom one_dim_random_walker.core.states.state import IntegerState"
#               "\nfrom one_dim_random_walker.core.states.state_space import IntegerLine\n"
#               "initial_state = IntegerState(0)"
#               "\nfield_norm_transform = RealFieldNormTransform()\n"
#               "current_field=RealField.from_length(100,initial_state)")
# TEST_CODE = "transformed_field=field_norm_transform.apply(current_field,current_field)"
# #added_field.plot_field()
# times = timeit.repeat(
#     stmt=TEST_CODE,
#     setup=SETUP_CODE,
#     number=number_of_executions,
#     repeat=number_of_repeats
# )
# best_total_time = min(times)
#
# #Calculate the average time per method call
# average_time_per_call = best_total_time / number_of_executions
# print("Testing for the field norm")
# print(f"Total test time (best of {number_of_repeats} repeats): {best_total_time:.4f} seconds")
# print(f"Average time per call ({number_of_executions} runs): {average_time_per_call * 1000000:.3f} microseconds")

# -------------- Single Step -------------------- #

single_step_test = OneDimSingleStep()
system_field_ref = RealField.from_length(1000,initial_state)
single_step_test.build_single_step_field(initial_state,system_field_ref)
# single_step_test.build_single_step_field_new(initial_state,system_field_ref).plot_field()
#
# SETUP_CODE = ("from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField, RealFieldAddition,RealFieldAdditionStateless,RealFieldNormTransform"
#               "\nfrom one_dim_random_walker.core.states.multi_step_reaching import OneDimensionMultiStepReaching"
#               "\nfrom one_dim_random_walker.core.states.reachable import OneDimensionReachable"
#               "\nfrom one_dim_random_walker.core.states.state import IntegerState"
#               "\nfrom one_dim_random_walker.core.states.state_space import IntegerLine\nfrom one_dim_random_walker.dynamics.single_step import OneDimSingleStep\n"
#               "initial_state = IntegerState(0)"
#               "\nsingle_step_test = OneDimSingleStep()\nsystem_field_ref = RealField.from_length(1000,initial_state)\n"
#               "")
# TEST_CODE = "single_step_test.build_single_step_field_new(initial_state,system_field_ref)"
# #added_field.plot_field()
# times = timeit.repeat(
#     stmt=TEST_CODE,
#     setup=SETUP_CODE,
#     number=number_of_executions,
#     repeat=number_of_repeats
# )
# best_total_time = min(times)
#
# #Calculate the average time per method call
# average_time_per_call = best_total_time / number_of_executions
# print("Testing for the single step")
# print(f"Total test time (best of {number_of_repeats} repeats): {best_total_time:.4f} seconds")
# print(f"Average time per call ({number_of_executions} runs): {average_time_per_call * 1000000:.3f} microseconds")

# before optimization

#---------------- Multi_step -----------------#


one_dim_multi_step =  OneDimRandomWalkerMultiStep(distance=1)

space = IntegerLine(current_state=initial_state,states_list=[initial_state])

single_transformed_field_proto = RealField.from_length(1,initial_state)
single_step_transformed_field_proto=RealField.from_length(1,initial_state)
prev_field_input = RealField(space)
prev_field_input.set_zero_field()
prev_field_input.set_field(initial_state,RealFieldValue(RealSingleFieldValue(1)))


multi_field=one_dim_multi_step.generate_multi_step_field(space,initial_state,3,single_transformed_field_proto,single_step_transformed_field_proto,
                                              prev_field_input )
# multi_field.plot_field()
#----------- Operator ------------------#

#
# print("Expectation_operator: " , OneDimRandomWalkerExpectationOperator().get_next_state(multi_field) )
# print("Probabiliostic_operator: " , OneDimRandomWalkerSampleOperator().get_next_state(multi_field) )
#

# -----------Dynamic System ----------------#

sys_field=RealField.from_length(100,initial_state)

sys_field.set_unit_field_at_state(initial_state)

#sys_field.plot_field()



one_dim_walker_fds  = OneDimWalkerFieldDynamicSystem(initial_state,sys_field,OneDimRandomWalkerMultiStep(2),OneDimensionReachable(2),OneDimRandomWalkerSampleOperator())
# print(one_dim_walker_fds.transition_list)
# one_dim_walker_fds.evolve(2)
# print(one_dim_walker_fds.field.get_field(one_dim_walker_fds.initial_state).data.value)
# print(one_dim_walker_fds.transition_list)
# one_dim_walker_fds.evolve(13)
# print(one_dim_walker_fds.field.get_field(one_dim_walker_fds.initial_state).data.value)
# print(one_dim_walker_fds.transition_list)
# one_dim_walker_fds.evolve(13)
# print(one_dim_walker_fds.field.get_field(one_dim_walker_fds.initial_state).data.value)
# print(one_dim_walker_fds.transition_list)
# print(one_dim_walker_fds.field.get_field(one_dim_walker_fds.initial_state).data.value)


# ------------------- Is Correlated ----------------------#
sys1_initial_stata=IntegerState(0)
sys2_initial_state=IntegerState(1)
sys3_initial_state=IntegerState(3)

one_dim_walker_fds_1  = OneDimWalkerFieldDynamicSystem(sys1_initial_stata,sys_field,OneDimRandomWalkerMultiStep(1),OneDimensionReachable(1),OneDimRandomWalkerSampleOperator())

one_dim_walker_fds_2 = OneDimWalkerFieldDynamicSystem(sys2_initial_state,sys_field,OneDimRandomWalkerMultiStep(1),OneDimensionReachable(1),OneDimRandomWalkerSampleOperator())

one_dim_walker_fds_3 = OneDimWalkerFieldDynamicSystem(sys3_initial_state,sys_field,OneDimRandomWalkerMultiStep(1),OneDimensionReachable(1),OneDimRandomWalkerSampleOperator())


dict_of_systems = {"sys1":one_dim_walker_fds_1,"sys2":one_dim_walker_fds_2,"sys3":one_dim_walker_fds_3 }
objIsRWCorrelated = IsRWCorrelated(1)

obj_correlationAffectingGroups  = CorrelationAffectingGroups(dict_of_systems,objIsRWCorrelated)
#print(obj_correlationAffectingGroups.get_affected_groups_dict())

initil_state_dict = {"sys1": sys1_initial_stata, "sys2": sys2_initial_state, "sys3": sys3_initial_state}

correlated_initial_state = CorrelatedState(initil_state_dict)
#print(correlated_initial_state.get_all_components())

#-------------- Affcting reachable for Correlated Random Walker--------#


obj_affecting_reachable =   RandomWalkerCorrelatedReachable(dict_of_systems)

#print(obj_affecting_reachable.get_reachable(correlated_initial_state))

#-------- Affecting Single Step View --------------#
one_dim_correlated_single_step = RandomWalkerCorrelatedSingleStep(dict_of_systems)
#
system_field = RealField.from_length(1,initial_state)
# #
# start_time = time.time()
# temp_single_step = one_dim_correlated_single_step.build_single_step_field(correlated_initial_state,system_field)
# end_time = time.time()
# print("Execution Time single-step build", end_time-start_time)
# #
#
# temp_single_step.plot_field();end_time = time.time()

#---------- Random Walker Correlated MultiStep ---------#

"""def generate_multi_step_field(
            self,
            space: "StatSpace[S]",  # ambient/type anchor
            start: S,
            L: int,
            single_transformed_field_proto: "Field[S]" = None,
            single_step_transformed_field_proto: "Field[S]" = None,
            prev_field_input: "Field[S]" = None,  # may be mutated (as you allowed)
            global_field: "Field[S]" = None"""
obj_random_walker_correlated_multi_step_field = RandomWalkerCorrelatedMultiStepField(dict_of_systems=dict_of_systems)

multi_step_space_corr = obj_affecting_reachable.get_reachable(correlated_initial_state)
start_corr = correlated_initial_state
L=2
single_transformed_field_proto = RealField.build_value_from_state_space(RealField,multi_step_space_corr,RealFieldValue.from_value(1))
single_step_transformed_field_proto = RealField.build_value_from_state_space(RealField,multi_step_space_corr,RealFieldValue.from_value(1))
prev_field_input_corr = RealField.build_value_from_state_space(RealField,multi_step_space_corr,RealFieldValue.from_value(0))
prev_field_input_corr.set_unit_field_at_state(correlated_initial_state)
gloabl_field = RealField.build_value_from_state_space(RealField,multi_step_space_corr,RealFieldValue.from_value(0))
#
# start_time = time.time()
# multi_step_field_correlated_v1 = obj_random_walker_correlated_multi_step_field.generate_multi_step_field(multi_step_space_corr,
#                                                                         start_corr,L,single_transformed_field_proto,
#                                                                         single_step_transformed_field_proto,
#                                                                         prev_field_input_corr,
#                                                                         gloabl_field)
# end_time = time.time()
# print("Total time taken for multi_step_correlated: ", end_time-start_time)
# print(type(multi_step_field_correlated_v1.state_space))
# multi_step_field_correlated_v1.plot_field('C')

#---------- Grouped Operator ------------#



#------------ Affecting FDS ----------#

one_dim_rw_correlated_fds =   OneDimRWCorrelatedAffectingFDS(dict_of_systems)
#
# one_dim_rw_correlated_fds.save_multi_step_field(3)
# print(one_dim_rw_correlated_fds.initial_state)
# one_dim_rw_correlated_fds.evolve_from_field()
# print(one_dim_rw_correlated_fds.initial_state)

# ----------------- Grouped Evolution ----------#

grouped_evolution=  OneDimRWCorrelatedGroupEvolution(dict_of_systems)
grouped_evolution.generate_affecting_groups()

print("initial state ", grouped_evolution.get_states_of_systems())

grouped_evolution.evolve(2)
print("final states after evolution: ", grouped_evolution.get_states_of_systems())
